"""
Edge / structural membership probe for GNN prediction APIs.

Black-box score adversary: for candidate edges (u,v), score = cosine similarity
of released posteriors p_u, p_v (plus optional confidence product). Trained edges
vs random non-edges → AUROC. Complements node LiRA under a wider threat model.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def edge_membership_auc(
    probs: np.ndarray,
    edge_index: np.ndarray,
    n_nodes: int,
    n_neg: Optional[int] = None,
    seed: int = 0,
) -> Tuple[float, dict]:
    p = np.asarray(probs, dtype=float)
    ei = np.asarray(edge_index)
    # Unique undirected positive edges
    a = np.minimum(ei[0], ei[1])
    b = np.maximum(ei[0], ei[1])
    pos = np.unique(np.stack([a, b], axis=1), axis=0)
    pos = pos[pos[:, 0] != pos[:, 1]]
    n_pos = len(pos)
    if n_pos < 20:
        return float("nan"), {"n_pos": n_pos, "n_neg": 0}
    n_neg = int(n_neg) if n_neg is not None else n_pos
    rng = np.random.RandomState(int(seed))
    pos_set = set(map(tuple, pos.tolist()))
    neg = []
    trials = 0
    while len(neg) < n_neg and trials < n_neg * 50:
        trials += 1
        u, v = int(rng.randint(0, n_nodes)), int(rng.randint(0, n_nodes))
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in pos_set:
            continue
        neg.append(key)
    if len(neg) < 20:
        return float("nan"), {"n_pos": n_pos, "n_neg": len(neg)}

    def score(u, v):
        pu, pv = p[u], p[v]
        # Cosine similarity of posteriors + confidence product
        cos = float(np.dot(pu, pv) / (np.linalg.norm(pu) * np.linalg.norm(pv) + 1e-12))
        return cos * float(pu.max() * pv.max())

    s_pos = np.array([score(u, v) for u, v in pos[:n_pos]], dtype=float)
    s_neg = np.array([score(u, v) for u, v in neg], dtype=float)
    y = np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))])
    s = np.concatenate([s_pos, s_neg])
    auc = float(roc_auc_score(y, s))
    return auc, {"n_pos": int(len(s_pos)), "n_neg": int(len(s_neg)), "mean_pos": float(s_pos.mean()), "mean_neg": float(s_neg.mean())}
