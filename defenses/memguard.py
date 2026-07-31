"""
Hardened MemGuard-style release (closer to Jia et al., CCS 2019).

Uses an MLP attack on φ-features and iterative L1-ball search that preserves
argmax. Faster than per-coordinate FD: random directional probes + occasional
coordinate steps.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


def _phi(probs: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    conf = p.max(axis=1)
    ent = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)
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
    use_mlp: bool = True,
):
    x = np.vstack([
        _phi(member_probs, member_labels),
        _phi(nonmember_probs, nonmember_labels),
    ])
    y = np.concatenate([
        np.ones(len(member_probs), dtype=int),
        np.zeros(len(nonmember_probs), dtype=int),
    ])
    if len(np.unique(y)) < 2:
        clf = LogisticRegression(max_iter=200)
        clf.classes_ = np.array([0, 1])
        clf.coef_ = np.zeros((1, x.shape[1]))
        clf.intercept_ = np.zeros(1)
        return clf
    if use_mlp:
        clf = MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=300,
            random_state=0,
            early_stopping=True,
            validation_fraction=0.15,
        )
        try:
            clf.fit(x, y)
            return clf
        except Exception:
            pass
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(x, y)
    return clf


def _member_score(clf, p_row: np.ndarray) -> float:
    phi = _phi(p_row.reshape(1, -1))
    proba = clf.predict_proba(phi)
    classes = list(getattr(clf, "classes_", [0, 1]))
    if 1 in classes:
        return float(proba[0, classes.index(1)])
    return float(proba[0, -1])


def _project(v: np.ndarray, top: int, base: np.ndarray, max_l1: float) -> np.ndarray:
    v = np.clip(v, 0.0, None)
    s = v.sum()
    v = v / s if s > 0 else np.ones_like(v) / len(v)
    if int(v.argmax()) != top:
        v[top] = v.max() + 1e-3
        v = np.clip(v, 0.0, None)
        v = v / v.sum()
    delta = v - base
    l1 = np.abs(delta).sum()
    if l1 > max_l1 and l1 > 0:
        v = base + delta * (max_l1 / l1)
        v = np.clip(v, 0.0, None)
        v = v / v.sum()
        if int(v.argmax()) != top:
            v[top] = v.max() + 1e-3
            v = v / v.sum()
    return v


def memguard_perturb(
    probs: np.ndarray,
    attack_clf,
    max_l1: float = 0.2,
    n_steps: int = 40,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    p0 = np.asarray(probs, dtype=float).copy()
    n, c = p0.shape
    out = p0.copy()
    for i in range(n):
        base = p0[i].copy()
        top = int(base.argmax())
        best = base.copy()
        best_score = _member_score(attack_clf, best)
        cur = best.copy()
        for _ in range(int(n_steps)):
            direction = rng.normal(0.0, 1.0, size=c)
            direction /= (np.linalg.norm(direction) + 1e-12)
            # Try both signs; keep the one that lowers member score
            step = max_l1 / max(4.0, float(c))
            improved = False
            for sign in (-1.0, 1.0):
                cand = _project(cur + sign * step * direction, top, base, max_l1)
                score = _member_score(attack_clf, cand)
                if score < best_score:
                    best_score = score
                    best = cand
                    cur = cand
                    improved = True
                    break
            if not improved and rng.rand() < 0.2:
                cand = _project(base + rng.laplace(0.0, max_l1 / c, size=c), top, base, max_l1)
                score = _member_score(attack_clf, cand)
                if score < best_score:
                    best_score = score
                    best = cand
                    cur = cand
        out[i] = best
    return out


def apply_memguard(
    probs: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    labels: np.ndarray,
    max_l1: float = 0.2,
    seed: int = 0,
    n_steps: int = 40,
) -> Tuple[np.ndarray, dict]:
    trm = np.asarray(train_mask, dtype=bool)
    tem = np.asarray(test_mask, dtype=bool)
    yn = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    clf = fit_attack_classifier(p[trm], p[tem], yn[trm], yn[tem], use_mlp=True)
    out = memguard_perturb(p, clf, max_l1=max_l1, n_steps=n_steps, seed=seed)
    return out, {
        "noise_mass": float(np.abs(out - p).sum()),
        "frac_protected": 1.0,
        "mean_scale": float(np.abs(out - p).mean()),
        "protector": "memguard_iter",
        "max_l1": float(max_l1),
        "n_steps": int(n_steps),
    }
