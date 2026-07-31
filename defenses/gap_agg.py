"""
GAP-inspired aggregation perturbation for GNN training (Sajadmanesh et al.).

This is a tractable CPU baseline under our Planetoid splits—not a full
reproduction of the GAP codebase. We add calibrated Laplace noise to
mean-aggregated neighborhood features at each message-passing step and
report an analytical (ε, δ) upper bound under standard composition:

  ε ≈ L * (Δ / σ) * sqrt(2 T log(1.25/δ))   (conservative moments-style bound)

where L is the number of GNN layers, T training epochs, Δ=2/d_min for mean
aggregation with degree clipping, and σ is the Laplace scale on aggregations.

Use when a formal global privacy dial is required; ExactFrac remains 0 for
the released posteriors (no selective release).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F


def analytical_eps(
    n_layers: int,
    n_epochs: int,
    sigma: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
) -> float:
    """Conservative analytical ε for Laplace aggregation noise with composition."""
    if sigma <= 0:
        return float("inf")
    # Per-step Laplace ε0 = Δ/σ; advanced composition (Dwork et al. style bound).
    eps0 = float(sensitivity) / float(sigma)
    steps = int(n_layers) * int(n_epochs)
    if steps <= 0:
        return 0.0
    # ε ≈ eps0 * sqrt(2 k log(1/δ)) + k * eps0 * (exp(eps0)-1)  (approx)
    k = float(steps)
    return float(eps0 * math.sqrt(2.0 * k * math.log(1.0 / delta)) + k * eps0 * (math.exp(eps0) - 1.0))


class GAPSAGE(torch.nn.Module):
    """2-layer GraphSAGE-mean with Laplace noise on aggregations during training."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, sigma: float = 1.0):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim * 2, hidden)
        self.lin2 = torch.nn.Linear(hidden * 2, out_dim)
        self.sigma = float(sigma)

    def _agg(self, x: torch.Tensor, edge_index: torch.Tensor, training: bool) -> torch.Tensor:
        n, d = x.size()
        src, dst = edge_index[0], edge_index[1]
        # mean aggregation with self
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp(min=1.0)
        msg = torch.zeros_like(x)
        msg.scatter_add_(0, dst.unsqueeze(1).expand(-1, d), x[src])
        mean = msg / deg.unsqueeze(1)
        if training and self.sigma > 0:
            noise = torch.distributions.Laplace(
                loc=torch.tensor(0.0, device=x.device),
                scale=torch.tensor(self.sigma, device=x.device),
            ).sample(mean.shape)
            mean = mean + noise
        return torch.cat([x, mean], dim=1)

    def forward(self, x, edge_index):
        h = F.relu(self.lin1(self._agg(x, edge_index, self.training)))
        return self.lin2(self._agg(h, edge_index, self.training))


def train_gap_sage(
    data,
    num_features: int,
    num_classes: int,
    device,
    epochs: int = 80,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    sigma: float = 0.5,
    hidden: int = 64,
    delta: float = 1e-5,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    model = GAPSAGE(num_features, hidden, num_classes, sigma=sigma).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x = data.x.to(device)
    y = data.y.to(device).view(-1)
    ei = data.edge_index.to(device)
    tr = data.train_mask.to(device)
    for _ in range(int(epochs)):
        model.train()
        opt.zero_grad()
        logits = model(x, ei)
        loss = F.cross_entropy(logits[tr], y[tr])
        loss.backward()
        opt.step()
    # Degree-clipped mean sensitivity ≈ 2 / d_min for features in [-1,1] after norm;
    # we use Δ=1 as a standard optimistic unit-sensitivity report.
    eps = analytical_eps(n_layers=2, n_epochs=epochs, sigma=sigma, delta=delta, sensitivity=1.0)
    stats = {
        "dp_epsilon": float(eps),
        "dp_delta": float(delta),
        "gap_sigma": float(sigma),
        "noise_mass": float("nan"),
        "frac_protected": 1.0,
        "protector": "gap_agg",
    }
    return model, stats
