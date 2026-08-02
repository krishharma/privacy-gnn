"""
HARP: Hop-Aware selective Release for ExactFrac-constrained score APIs.

Framework (not a new noise distribution):
  1. Rank nodes with a pluggable constructor (topology / random / degree /
     confidence / audit / oracle).
  2. Select seeds and expand by k hops so aggregation cannot reintroduce
     cues through unprotected neighbors.
  3. Apply a pluggable protector (Laplace or masking) only on the protected
     set; the complement stays bit-exact (ExactFrac = 1 - Frac).
  4. Optionally run Constrained Frac Search (CFS) to maximize Acc subject to
     ExactFrac >= c and LiRA <= tau.

Default paper configuration is release-only (no train-time alignment).
Topology LTE is a cheap default constructor, not the claimed novelty.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

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


def risk_from_degree(data, invert: bool = True) -> torch.Tensor:
    """Degree (or inverse-degree) constructor: prefer low-degree / high-degree nodes."""
    n = int(data.num_nodes)
    ei = data.edge_index
    if torch.is_tensor(ei):
        src = ei[0].detach().cpu().numpy().astype(np.int64)
        dst = ei[1].detach().cpu().numpy().astype(np.int64)
    else:
        src = np.asarray(ei[0], dtype=np.int64)
        dst = np.asarray(ei[1], dtype=np.int64)
    deg = np.zeros(n, dtype=float)
    for u, v in zip(src, dst):
        if 0 <= u < n and 0 <= v < n:
            deg[u] += 1.0
            deg[v] += 1.0
    # Undirected edges are double-counted; keep relative ranks.
    r = 1.0 / (deg + 1.0) if invert else deg
    if r.max() > r.min():
        r = (r - r.min()) / (r.max() - r.min() + 1e-12)
    return torch.tensor(r, dtype=torch.float)


def risk_from_train_neighbors(data) -> torch.Tensor:
    """Fraction of neighbors in the supervised train set (train-exposure)."""
    n = int(data.num_nodes)
    ei = data.edge_index
    if torch.is_tensor(ei):
        src = ei[0].detach().cpu().numpy().astype(np.int64)
        dst = ei[1].detach().cpu().numpy().astype(np.int64)
    else:
        src = np.asarray(ei[0], dtype=np.int64)
        dst = np.asarray(ei[1], dtype=np.int64)
    tr = data.train_mask.detach().cpu().numpy().astype(bool)
    adj = [[] for _ in range(n)]
    for u, v in zip(src, dst):
        if 0 <= u < n and 0 <= v < n:
            adj[u].append(int(v))
            adj[v].append(int(u))
    r = np.zeros(n, dtype=float)
    for u in range(n):
        nbrs = adj[u]
        if not nbrs:
            r[u] = float(tr[u])
            continue
        r[u] = float(np.mean([tr[v] for v in nbrs]))
    if r.max() > r.min():
        r = (r - r.min()) / (r.max() - r.min() + 1e-12)
    return torch.tensor(r, dtype=torch.float)


def risk_from_confidence(probs: np.ndarray, mode: str = "maxconf") -> torch.Tensor:
    """
    Score-based constructor from clean posteriors.
    mode='maxconf': protect high-confidence nodes (classic membership cue).
    mode='entropy': protect high-entropy (uncertain) nodes.
    """
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    if mode == "entropy":
        r = -np.sum(p * np.log(p), axis=1)
    else:
        r = p.max(axis=1)
    if r.max() > r.min():
        r = (r - r.min()) / (r.max() - r.min() + 1e-12)
    return torch.tensor(r, dtype=torch.float)


def risk_ensemble(*risks: torch.Tensor, weights: Optional[Sequence[float]] = None) -> torch.Tensor:
    """Weighted average of normalized risk constructors (audit+entropy etc.)."""
    rs = [torch.as_tensor(r, dtype=torch.float).reshape(-1) for r in risks]
    if not rs:
        raise ValueError("risk_ensemble requires at least one risk vector")
    if weights is None:
        weights = [1.0] * len(rs)
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    acc = torch.zeros_like(rs[0])
    for wi, r in zip(w, rs):
        rr = r.clone()
        if float(rr.max() - rr.min()) > 1e-12:
            rr = (rr - rr.min()) / (rr.max() - rr.min() + 1e-12)
        acc = acc + float(wi) * rr
    return acc


def slice_constrained_frac_search(
    evaluate_frac: Callable[[float], Dict[str, float]],
    exact_frac_min: float,
    slice_auc_max: float,
    lira_max: Optional[float] = None,
    frac_grid: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Slice-aware CFS: maximize Acc s.t. ExactFrac>=c and clean-slice conf AUROC<=τ_s
    (optional LiRA<=τ). Raises Frac (lowers ExactFrac toward c) until the clean
    majority is no longer an easy membership pocket.
    """
    c = float(exact_frac_min)
    tau_s = float(slice_auc_max)
    if frac_grid is None:
        max_frac = max(0.0, 1.0 - c)
        frac_grid = [round(x, 3) for x in np.linspace(0.0, max_frac, 9)]
    rows: List[Dict[str, float]] = []
    for f in frac_grid:
        if float(f) > 1.0 - c + 1e-9:
            continue
        out = dict(evaluate_frac(float(f)))
        out["Frac"] = float(f)
        out["ExactFrac_design"] = 1.0 - float(f)
        rows.append(out)
    if not rows:
        return {"feasible": 0.0, "Frac": float("nan")}
    ok = []
    for r in rows:
        if float(r.get("slice_auc", 1.0)) > tau_s + 1e-12:
            continue
        if lira_max is not None and float(r.get("LiRA", 1.0)) > float(lira_max) + 1e-12:
            continue
        ok.append(r)
    if ok:
        best = max(ok, key=lambda r: float(r.get("Acc", -1.0)))
        best["feasible_slice"] = 1.0
    else:
        # Prefer lowest slice AUC among ExactFrac-feasible points.
        best = min(rows, key=lambda r: float(r.get("slice_auc", 1.0)))
        best["feasible_slice"] = 0.0
    best["feasible_exact"] = 1.0
    best["c"] = c
    best["tau_slice"] = tau_s
    return best


def constrained_frac_search(
    evaluate_frac: Callable[[float], Dict[str, float]],
    exact_frac_min: float,
    lira_max: float,
    frac_grid: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Constrained Frac Search (CFS).

    ExactFrac = 1 - Frac, so ExactFrac >= c implies Frac <= 1 - c.
    Among Frac in the feasible grid, return the point with maximum Acc
    among those with LiRA <= lira_max. If none meet the audit, return the
    feasible point with lowest LiRA (and mark feasible_audit=0).

    evaluate_frac(frac) must return dict with keys Acc, LiRA, ExactFrac, Mass.
    """
    c = float(exact_frac_min)
    tau = float(lira_max)
    if frac_grid is None:
        # Frac from 0 to 1-c inclusive.
        max_frac = max(0.0, 1.0 - c)
        frac_grid = [round(x, 3) for x in np.linspace(0.0, max_frac, 9)]
    rows: List[Dict[str, float]] = []
    for f in frac_grid:
        if float(f) > 1.0 - c + 1e-9:
            continue
        out = dict(evaluate_frac(float(f)))
        out["Frac"] = float(f)
        out["ExactFrac_design"] = 1.0 - float(f)
        rows.append(out)
    if not rows:
        return {"feasible": 0.0, "Frac": float("nan"), "Acc": float("nan"), "LiRA": float("nan")}
    audit_ok = [r for r in rows if float(r.get("LiRA", 1.0)) <= tau + 1e-12]
    if audit_ok:
        best = max(audit_ok, key=lambda r: float(r.get("Acc", -1.0)))
        best["feasible_audit"] = 1.0
    else:
        best = min(rows, key=lambda r: float(r.get("LiRA", 1.0)))
        best["feasible_audit"] = 0.0
    best["feasible_exact"] = 1.0
    best["c"] = c
    best["tau"] = tau
    return best


# Legacy locked config (train-time alignment). Prefer LOCKED_HARP_RELEASE for paper.
LOCKED_HARP = {
    "lam": 0.5,
    "use_lte": True,
    "use_gate": True,
    "arch_aware": True,
    "risk_frac": 0.30,
    "k_hops": 1,
    "strong_noise_scale": 0.30,
    "weak_noise_scale": 0.0,
    "target_protect_frac": 0.40,
    "budget_B": 0.0,
    "warmup_epochs": 5,
    "entropy_coef": 0.05,
    "train_on_protected": True,
}

def deterministic_confidence_smooth(
    probs: np.ndarray,
    clean_mask: np.ndarray,
    theta: float = 0.90,
    temp: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replay-stable, argmax-preserving temperature flatten on high-confidence
    clean-slice rows. Deterministic ⇒ ExactFrac and caches are preserved.
    """
    p2 = np.asarray(probs, dtype=float).copy()
    mask = np.asarray(clean_mask, dtype=bool).reshape(-1)
    hot = mask & (p2.max(axis=1) > float(theta))
    if hot.any():
        logp = np.log(np.clip(p2[hot], 1e-12, 1.0)) / float(temp)
        e = np.exp(logp - logp.max(axis=1, keepdims=True))
        p2[hot] = e / e.sum(axis=1, keepdims=True)
    return p2, hot


# Paper-default: release-only selective Laplace under ExactFrac SLA (c=0.60 → Frac=0.40).
LOCKED_HARP_RELEASE = {
    "lam": 0.0,
    "use_lte": True,
    "use_gate": False,
    "arch_aware": True,
    "risk_frac": 0.30,
    "k_hops": 1,
    "strong_noise_scale": 0.30,
    "weak_noise_scale": 0.0,
    "target_protect_frac": 0.40,
    "budget_B": 0.0,
    "warmup_epochs": 0,
    "entropy_coef": 0.0,
    "train_on_protected": False,
    "seed_mode": "lte",
}
