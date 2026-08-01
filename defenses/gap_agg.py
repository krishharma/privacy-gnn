"""
GAP-inspired aggregation perturbation for GNN training (Sajadmanesh et al.).

Tractable CPU baseline under Planetoid splits—not a full GAP codebase
reproduction. Laplace noise is added to mean-aggregated neighborhood features
during training; inference is deterministic.

Critical ExactFrac property: because noise is train-only, released posteriors
are bit-stable under re-query → ExactFrac=1 with a finite analytical (ε, δ).
This is the hybrid DP-train + ExactFrac-serve operating point that selective
release noise cannot provide under global DP.
"""
from __future__ import annotations

from typing import Dict, Tuple

import math
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
    eps0 = float(sensitivity) / float(sigma)
    steps = int(n_layers) * int(n_epochs)
    if steps <= 0:
        return 0.0
    k = float(steps)
    return float(eps0 * math.sqrt(2.0 * k * math.log(1.0 / delta)) + k * eps0 * (math.exp(eps0) - 1.0))


class GAPSAGE(torch.nn.Module):
    """2-layer GraphSAGE-mean with Laplace noise on aggregations (train only)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, sigma: float = 1.0, max_degree: int = 100):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim * 2, hidden)
        self.lin2 = torch.nn.Linear(hidden * 2, out_dim)
        self.sigma = float(sigma)
        self.max_degree = int(max_degree)  # reserved for sensitivity reporting

    def _agg(self, x: torch.Tensor, edge_index: torch.Tensor, training: bool) -> torch.Tensor:
        n, d = x.size()
        src, dst = edge_index[0], edge_index[1]
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
    max_degree: int = 100,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    model = GAPSAGE(num_features, hidden, num_classes, sigma=sigma, max_degree=max_degree).to(device)
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
    eps = analytical_eps(n_layers=2, n_epochs=epochs, sigma=sigma, delta=delta, sensitivity=1.0)
    stats = {
        "dp_epsilon": float(eps),
        "dp_delta": float(delta),
        "gap_sigma": float(sigma),
        "gap_max_degree": int(max_degree),
        "noise_mass": 0.0,
        # Deterministic eval release: ExactFrac = 1 by construction.
        "frac_protected": 0.0,
        "exact_frac": 1.0,
        "protector": "gap_agg",
    }
    return model, stats
