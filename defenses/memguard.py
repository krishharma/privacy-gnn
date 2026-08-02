"""
MemGuard-style release (Jia et al., CCS 2019), hardened for our evaluation.

Fits a shadow-style φ-attack (confidence / entropy / margin) and searches an
L1 ball that preserves argmax so the attack's P(member) is driven toward 0.5.
We report both the φ-attack MemGuard optimizes against and LiRA (which it is
not designed to resist), following Aerni et al. (CCS 2024).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _phi(probs: np.ndarray) -> np.ndarray:
    """Label-free φ used at both fit and release time."""
    p = np.asarray(probs, dtype=float)
    conf = p.max(axis=1)
    ent = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)
    part = np.partition(p, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    return np.stack([conf, ent, margin], axis=1)


def fit_attack_classifier(
    member_probs: np.ndarray,
    nonmember_probs: np.ndarray,
    use_mlp: bool = True,
):
    x = np.vstack([_phi(member_probs), _phi(nonmember_probs)])
    y = np.concatenate([
        np.ones(len(member_probs), dtype=int),
        np.zeros(len(nonmember_probs), dtype=int),
    ])
    if len(np.unique(y)) < 2:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
        # degenerate: constant predictor
        clf.fit(x, np.array([0, 1] * (len(x) // 2) + [0] * (len(x) % 2))[: len(x)])
        return clf

    candidates = []
    if use_mlp:
        candidates.append(
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=1500,
                    random_state=0,
                    early_stopping=False,
                ),
            )
        )
    candidates.append(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"))
    )
    best, best_auc = None, -1.0
    for clf in candidates:
        try:
            clf.fit(x, y)
            auc = float(roc_auc_score(y, clf.predict_proba(x)[:, 1]))
            if auc > best_auc:
                best, best_auc = clf, auc
        except Exception:
            continue
    return best


def _make_fast_scorer(clf):
    """Pure-numpy P(member) from a StandardScaler+MLP/LR pipeline."""
    steps = dict(clf.named_steps)
    scaler = steps["standardscaler"]
    est = list(steps.values())[-1]
    mu, sd = scaler.mean_, scaler.scale_
    classes = list(est.classes_)
    pos = classes.index(1) if 1 in classes else -1

    if isinstance(est, MLPClassifier):
        coefs, icepts = est.coefs_, est.intercepts_
        out_dim = int(coefs[-1].shape[1]) if coefs[-1].ndim == 2 else 1

        def score(phi_row: np.ndarray) -> float:
            h = (phi_row - mu) / sd
            for w, b in zip(coefs[:-1], icepts[:-1]):
                h = np.maximum(h @ w + b, 0.0)
            logits = np.asarray(h @ coefs[-1] + icepts[-1], dtype=float).reshape(-1)
            if out_dim == 1 or logits.size == 1:
                p1 = 1.0 / (1.0 + np.exp(-float(logits.ravel()[0])))
                return p1 if pos != 0 else 1.0 - p1
            logits = logits - logits.max()
            ex = np.exp(logits)
            proba = ex / ex.sum()
            return float(proba[pos])

        return score

    w, b = est.coef_[0], float(est.intercept_[0])

    def score(phi_row: np.ndarray) -> float:
        h = (phi_row - mu) / sd
        z = float(h @ w + b)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return p1 if pos != 0 else 1.0 - p1

    return score


def _phi_row(v: np.ndarray) -> np.ndarray:
    conf = float(v.max())
    ent = float(-np.sum(v * np.log(np.clip(v, 1e-12, 1.0))))
    part = np.partition(v, -2)
    margin = float(part[-1] - part[-2])
    return np.array([conf, ent, margin], dtype=float)


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
    max_l1: float = 0.3,
    n_steps: int = 60,
    seed: int = 0,
) -> np.ndarray:
    """Drive P(member) toward 0.5 under argmax + L1 constraints (Jia objective)."""
    rng = np.random.RandomState(int(seed))
    p0 = np.asarray(probs, dtype=float).copy()
    n, c = p0.shape
    out = p0.copy()
    score_fn = _make_fast_scorer(attack_clf)

    for i in range(n):
        base = p0[i].copy()
        top = int(base.argmax())
        best = base.copy()
        best_dist = abs(score_fn(_phi_row(best)) - 0.5)
        cur = best.copy()
        stall = 0
        for _ in range(int(n_steps)):
            direction = rng.normal(0.0, 1.0, size=c)
            direction /= (np.linalg.norm(direction) + 1e-12)
            step = max_l1 / max(3.0, float(c))
            improved = False
            for sign in (-1.0, 1.0):
                cand = _project(cur + sign * step * direction, top, base, max_l1)
                dist = abs(score_fn(_phi_row(cand)) - 0.5)
                if dist < best_dist - 1e-4:
                    best_dist = dist
                    best = cand
                    cur = cand
                    improved = True
                    stall = 0
                    break
            if not improved:
                # Coordinate-wise confidence flattening toward uniform-on-non-top
                cand = best.copy()
                j = int(rng.randint(0, c))
                if j == top:
                    continue
                cand[j] += step
                cand = _project(cand, top, base, max_l1)
                dist = abs(score_fn(_phi_row(cand)) - 0.5)
                if dist < best_dist - 1e-4:
                    best_dist = dist
                    best = cand
                    cur = cand
                    stall = 0
                else:
                    stall += 1
                    if stall >= 8:
                        break
            if best_dist <= 0.02:
                break
        out[i] = best
    return out


def apply_memguard(
    probs: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    labels: np.ndarray,
    max_l1: float = 0.3,
    seed: int = 0,
    n_steps: int = 60,
) -> Tuple[np.ndarray, dict]:
    trm = np.asarray(train_mask, dtype=bool)
    tem = np.asarray(test_mask, dtype=bool)
    p = np.asarray(probs, dtype=float)
    rng = np.random.RandomState(int(seed))
    tr_idx = np.where(trm)[0]
    te_idx = np.where(tem)[0]
    rng.shuffle(tr_idx)
    rng.shuffle(te_idx)
    tr_fit = tr_idx[: max(20, len(tr_idx) // 2)]
    te_fit = te_idx[: max(20, len(te_idx) // 2)]
    clf = fit_attack_classifier(p[tr_fit], p[te_fit], use_mlp=True)
    # Diagnose fit-time φ-AUROC
    x_fit = np.vstack([_phi(p[tr_fit]), _phi(p[te_fit])])
    y_fit = np.concatenate([np.ones(len(tr_fit)), np.zeros(len(te_fit))])
    fit_auc = float(roc_auc_score(y_fit, clf.predict_proba(x_fit)[:, 1]))
    out = memguard_perturb(p, clf, max_l1=max_l1, n_steps=n_steps, seed=seed)
    # Post-defense φ-AUROC on held-out halves
    tr_h = tr_idx[len(tr_fit) :]
    te_h = te_idx[len(te_fit) :]
    if len(tr_h) > 10 and len(te_h) > 10:
        x_h = np.vstack([_phi(out[tr_h]), _phi(out[te_h])])
        y_h = np.concatenate([np.ones(len(tr_h)), np.zeros(len(te_h))])
        post_auc = float(roc_auc_score(y_h, clf.predict_proba(x_h)[:, 1]))
    else:
        post_auc = float("nan")
    return out, {
        "noise_mass": float(np.abs(out - p).sum()),
        "frac_protected": 1.0,
        "exact_frac": 0.0,
        "mean_scale": float(np.abs(out - p).mean()),
        "protector": "memguard_shadow_mlp",
        "max_l1": float(max_l1),
        "n_steps": int(n_steps),
        "memguard_fit_phi_auc": fit_auc,
        "memguard_post_phi_auc": post_auc,
        "n_fit_members": int(len(tr_fit)),
        "n_fit_nonmembers": int(len(te_fit)),
    }
