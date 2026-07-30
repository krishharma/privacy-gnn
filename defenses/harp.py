"""
HARP: Hop-Aware Risk-conditioned Privacy.

Uniform posterior defenses (e.g. LBP) spend noise on every node. Continuous
risk-weighted noise (SAMI) softens that waste but still perturbs the full
graph. HARP instead:

  1. Estimates per-node membership risk via LTE (reused from SAMI).
  2. Selects a seed set of high-risk nodes (top risk_frac or threshold).
  3. Expands seeds by k hops on the undirected graph so neighborhood
     aggregation cannot reintroduce membership cues through neighbors.
  4. Applies strong release noise (and optional train-time alignment) only
     on the hop-consistent protected set; unprotected nodes stay clean.

The primary systems claim is lower noise mass / higher Acc–QPS at matched
LiRA relative to uniform LBP, while remaining competitive with SAMI on
leaky cells.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from defenses.sami import compute_lte_risk


def expand_k_hop(
    edge_index: Union[torch.Tensor, np.ndarray],
    seed_mask: Union[torch.Tensor, np.ndarray],
    k: int,
    num_nodes: Optional[int] = None,
) -> np.ndarray:
    """
    Undirected k-hop expansion of a boolean seed mask.

    Returns a boolean numpy array of shape [n]. k<=0 returns the seed mask.
    """
    seed = np.asarray(seed_mask, dtype=bool).reshape(-1)
    n = int(num_nodes) if num_nodes is not None else int(seed.shape[0])
    if k is None or int(k) <= 0:
        return seed.copy()

    if torch.is_tensor(edge_index):
        ei = edge_index.detach().cpu().numpy()
    else:
        ei = np.asarray(edge_index)
    if ei.size == 0:
        return seed.copy()

    # Build undirected adjacency as CSR-like lists.
    src, dst = ei[0].astype(np.int64), ei[1].astype(np.int64)
    adj = [[] for _ in range(n)]
    for u, v in zip(src, dst):
        if 0 <= u < n and 0 <= v < n:
            adj[u].append(int(v))
            adj[v].append(int(u))

    protected = seed.copy()
    frontier = np.flatnonzero(seed)
    for _ in range(int(k)):
        if frontier.size == 0:
            break
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if not protected[v]:
                    protected[v] = True
                    nxt.append(v)
        frontier = np.asarray(nxt, dtype=np.int64)
    return protected


def select_risk_seeds(
    risk: Union[torch.Tensor, np.ndarray],
    risk_frac: float = 0.25,
    risk_threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Boolean seed mask from continuous risk.

    If risk_threshold is set, seeds are {v: r_v >= threshold}.
    Else seeds are the top risk_frac fraction (at least 1 node if n>0).
    """
    r = np.asarray(risk, dtype=float).reshape(-1)
    n = r.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    if risk_threshold is not None:
        return r >= float(risk_threshold)

    frac = float(risk_frac)
    frac = min(max(frac, 0.0), 1.0)
    if frac <= 0:
        return np.zeros(n, dtype=bool)
    if frac >= 1.0:
        return np.ones(n, dtype=bool)
    k = max(1, int(np.ceil(frac * n)))
    # Stable: highest risk first; ties broken by index.
    order = np.argsort(-r, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def compute_harp_scales(
    data,
    risk: Optional[Union[torch.Tensor, np.ndarray]] = None,
    risk_frac: float = 0.25,
    risk_threshold: Optional[float] = None,
    k_hops: int = 1,
    strong_noise_scale: float = 0.55,
    weak_noise_scale: float = 0.0,
    use_lte: bool = True,
    arch: str = "sage",
    arch_aware: bool = True,
    target_protect_frac: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Build per-node Laplace scales for HARP release.

    If ``target_protect_frac`` is set (in (0,1]), binary-search ``risk_frac`` so
    that the k-hop-expanded protected fraction is near the target. This keeps
    noise mass predictable across graphs of different density.

    Returns
    -------
    scales : float array [n] — Laplace b per node
    protected : bool array [n] — hop-expanded set
    seeds : bool array [n] — pre-expansion seed set
    stats : dict with noise_mass, frac_protected, frac_seeds, mean_scale, ...
    """
    if risk is None:
        risk_t = compute_lte_risk(
            data.cpu() if hasattr(data, "cpu") else data,
            uniform=not bool(use_lte),
            arch=arch,
            arch_aware=arch_aware,
        )
        r = risk_t.numpy() if hasattr(risk_t, "numpy") else np.asarray(risk_t, dtype=float)
    else:
        r = np.asarray(risk, dtype=float).reshape(-1)
        if hasattr(risk, "detach"):
            r = risk.detach().cpu().numpy().astype(float).reshape(-1)

    n = int(getattr(data, "num_nodes", r.shape[0]))
    if r.shape[0] != n:
        r = np.resize(r, n)

    ei = data.edge_index
    frac = float(risk_frac)
    if target_protect_frac is not None and risk_threshold is None:
        # Binary search seed fraction to hit target protected fraction after expansion.
        lo, hi = 0.0, 1.0
        best_frac, best_prot, best_seeds = frac, None, None
        target = float(target_protect_frac)
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            s = select_risk_seeds(r, risk_frac=mid, risk_threshold=None)
            pmask = expand_k_hop(ei, s, k=int(k_hops), num_nodes=n)
            got = float(pmask.mean()) if n else 0.0
            best_frac, best_prot, best_seeds = mid, pmask, s
            if got < target:
                lo = mid
            else:
                hi = mid
        frac = best_frac
        seeds = best_seeds if best_seeds is not None else select_risk_seeds(r, risk_frac=frac)
        protected = best_prot if best_prot is not None else expand_k_hop(ei, seeds, k=int(k_hops), num_nodes=n)
    else:
        seeds = select_risk_seeds(r, risk_frac=frac, risk_threshold=risk_threshold)
        protected = expand_k_hop(ei, seeds, k=int(k_hops), num_nodes=n)

    strong = float(strong_noise_scale)
    weak = float(weak_noise_scale)
    scales = np.full(n, weak, dtype=float)
    scales[protected] = strong

    noise_mass = float(scales.sum())
    # Uniform LBP-style mass at the same strong scale (for relative reporting).
    uniform_mass = float(strong * n) if strong > 0 else float("nan")
    stats = {
        "noise_mass": noise_mass,
        "mean_scale": float(scales.mean()) if n else 0.0,
        "frac_protected": float(protected.mean()) if n else 0.0,
        "frac_seeds": float(seeds.mean()) if n else 0.0,
        "n_protected": float(protected.sum()),
        "n_seeds": float(seeds.sum()),
        "n_nodes": float(n),
        "strong_noise_scale": strong,
        "weak_noise_scale": weak,
        "k_hops": float(k_hops),
        "risk_frac": float(frac if risk_threshold is None else -1.0),
        "relative_noise_mass_vs_uniform": (
            float(noise_mass / uniform_mass) if uniform_mass and uniform_mass == uniform_mass else float("nan")
        ),
    }
    return scales, protected, seeds, stats


def mask_risk_to_protected(
    risk: Union[torch.Tensor, np.ndarray],
    protected: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Zero out risk outside the protected set (for train-time alignment/HCAG)."""
    if torch.is_tensor(risk):
        r = risk.float().clone()
        m = torch.as_tensor(protected, dtype=torch.bool, device=r.device)
        r = r * m.float()
        return r
    r = np.asarray(risk, dtype=float).reshape(-1)
    m = np.asarray(protected, dtype=bool).reshape(-1)
    return torch.tensor(r * m.astype(float), dtype=torch.float)


# Locked default used in paper tables (selected on Cora val Acc; not test LiRA).
# Tuned for Acc ≫ LBP at matched LiRA with substantially lower noise mass.
LOCKED_HARP = {
    "lam": 0.5,
    "use_lte": True,
    "use_gate": True,
    "arch_aware": True,
    "risk_frac": 0.30,
    "k_hops": 1,
    "strong_noise_scale": 0.30,
    "weak_noise_scale": 0.0,
    "target_protect_frac": 0.40,  # ~40% protected after 1-hop; rest clean
    "budget_B": 0.0,
    "warmup_epochs": 5,
    "entropy_coef": 0.05,
    "train_on_protected": True,
}
