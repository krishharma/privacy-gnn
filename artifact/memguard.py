"""
Simplified MemGuard-style release defense (Jia et al., CCS 2019).

Adds a utility-bounded adversarial perturbation to every posterior so that a
confidence-based attack classifier is confused while preserving the argmax
label. Unlike HARP, every response is perturbed (no clean majority).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


def _phi(probs: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    conf = p.max(axis=1)
    ent = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)
    # margin between top-1 and top-2
    part = np.partition(p, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    if labels is None:
        true_p = conf
    else:
        y = np.asarray(labels, dtype=int)
        true_p = p[np.arange(len(y)), y]
    return np.stack([conf, true_p, ent, margin], axis=1)


def fit_attack_classifier(
    member_probs: np.ndarray,
    nonmember_probs: np.ndarray,
    member_labels: Optional[np.ndarray] = None,
    nonmember_labels: Optional[np.ndarray] = None,
) -> LogisticRegression:
    x = np.vstack([
        _phi(member_probs, member_labels),
        _phi(nonmember_probs, nonmember_labels),
    ])
    y = np.concatenate([
        np.ones(len(member_probs), dtype=int),
        np.zeros(len(nonmember_probs), dtype=int),
    ])
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    if len(np.unique(y)) < 2:
        # Degenerate: return a no-op classifier
        clf.classes_ = np.array([0, 1])
        clf.coef_ = np.zeros((1, x.shape[1]))
        clf.intercept_ = np.zeros(1)
        return clf
    clf.fit(x, y)
    return clf


def memguard_perturb(
    probs: np.ndarray,
    attack_clf: LogisticRegression,
    max_l1: float = 0.2,
    n_steps: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """
    For each posterior, search a small simplex-preserving perturbation that
    keeps argmax and drives the attack classifier toward non-member (class 0).
    """
    rng = np.random.RandomState(int(seed))
    p0 = np.asarray(probs, dtype=float).copy()
    n, c = p0.shape
    out = p0.copy()
    # Finite-difference / random-search MemGuard (simplified, CPU-friendly).
    for i in range(n):
        base = p0[i].copy()
        top = int(base.argmax())
        best = base.copy()
        best_score = float(attack_clf.predict_proba(_phi(best.reshape(1, -1)))[0, 1])
        for _ in range(int(n_steps)):
            noise = rng.laplace(0.0, max_l1 / max(c, 1), size=c)
            cand = base + noise
            cand = np.clip(cand, 0.0, None)
            s = cand.sum()
            if s <= 0:
                continue
            cand = cand / s
            if int(cand.argmax()) != top:
                # restore argmax by boosting top class
                cand[top] = cand.max() + 1e-3
                cand = cand / cand.sum()
            score = float(attack_clf.predict_proba(_phi(cand.reshape(1, -1)))[0, 1])
            if score < best_score:
                best_score = score
                best = cand
        out[i] = best
    return out


def apply_memguard(
    probs: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    labels: np.ndarray,
    max_l1: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, dict]:
    """Fit attack on train vs test clean scores, then perturb all posteriors."""
    trm = np.asarray(train_mask, dtype=bool)
    tem = np.asarray(test_mask, dtype=bool)
    yn = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    clf = fit_attack_classifier(p[trm], p[tem], yn[trm], yn[tem])
    out = memguard_perturb(p, clf, max_l1=max_l1, seed=seed)
    stats = {
        "noise_mass": float(np.abs(out - p).sum()),
        "frac_protected": 1.0,
        "mean_scale": float(np.abs(out - p).mean()),
        "protector": "memguard",
    }
    return out, stats
