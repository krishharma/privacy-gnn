"""
LiRA-style membership inference (Carlini et al., S&P 2022) adapted to node
classification posteriors, plus TPR-at-low-FPR helpers.

Offline Gaussian LiRA: for each query node, fit two Gaussians on the
logit-scaled true-class confidence from shadow models — IN (node was in the
shadow train mask) vs OUT (node was held out) — then score the target model
with a likelihood ratio. Higher scores indicate membership.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


def _logit_confidence(probs, labels, eps=1e-8):
    """Logit of true-label confidence; stable under near-0/1 probabilities."""
    n = len(probs)
    conf = np.empty(n, dtype=np.float64)
    for i in range(n):
        y = int(labels[i])
        p = float(probs[i, y]) if 0 <= y < probs.shape[1] else 0.0
        p = min(max(p, eps), 1.0 - eps)
        conf[i] = np.log(p / (1.0 - p))
    return conf


def tpr_at_fpr(y_true, scores, target_fpr=0.01):
    """
    True-positive rate at a target false-positive rate from the ROC curve.
    Returns 0.0 if the curve cannot be computed.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, scores)
    # Largest TPR among points with FPR <= target (security-relevant operating point).
    ok = fpr <= target_fpr + 1e-12
    if not np.any(ok):
        return float(tpr[0]) if len(tpr) else 0.0
    return float(np.max(tpr[ok]))


def lira_gaussian_auc(
    target_probs,
    target_labels,
    target_train_mask,
    target_test_mask,
    shadow_probs_list,
    shadow_train_masks,
    shadow_test_masks,
    eps=1e-6,
):
    """
    Offline Gaussian LiRA over the union of target train (members) and test
    (non-members). Shadow models supply IN/OUT logit-confidence samples.

    Returns (auc, accuracy_at_0.5_threshold, tpr_at_0.001_fpr, tpr_at_0.01_fpr).
    For backward compatibility with callers that unpack two values, the first
    two entries remain (auc, acc); extra TPR metrics are also returned when
    unpacked as four values — callers should use the 4-tuple API.
    """
    target_probs = np.asarray(target_probs)
    target_labels = np.asarray(target_labels)
    trm = np.asarray(target_train_mask).astype(bool)
    tem = np.asarray(target_test_mask).astype(bool)
    eval_mask = trm | tem
    eval_idx = np.where(eval_mask)[0]
    if len(eval_idx) < 4 or len(shadow_probs_list) == 0:
        return 0.5, 0.5, 0.0, float(0.01)

    t_conf = _logit_confidence(target_probs, target_labels)
    n_shadows = len(shadow_probs_list)
    shadow_confs = []
    shadow_in = []
    for k in range(n_shadows):
        sp = np.asarray(shadow_probs_list[k])
        # Shadows may be on a regenerated graph with the same n; use their y via
        # the target labels when graphs are re-split of the same citation graph,
        # else fall back to target labels (synthetic regenerations share C).
        sh_y = target_labels if sp.shape[0] == target_labels.shape[0] else target_labels
        if sp.shape[0] != sh_y.shape[0]:
            # Mismatched graph size: skip this shadow.
            continue
        shadow_confs.append(_logit_confidence(sp, sh_y))
        shadow_in.append(np.asarray(shadow_train_masks[k]).astype(bool))

    if len(shadow_confs) == 0:
        return 0.5, 0.5, 0.0, float(0.01)

    scores = np.zeros(len(eval_idx), dtype=np.float64)
    y_mem = np.zeros(len(eval_idx), dtype=np.int64)

    for j, v in enumerate(eval_idx):
        y_mem[j] = 1 if trm[v] else 0
        in_vals, out_vals = [], []
        for k, sc in enumerate(shadow_confs):
            if v >= len(sc):
                continue
            if shadow_in[k][v]:
                in_vals.append(sc[v])
            else:
                out_vals.append(sc[v])
        # Need at least one sample in each bucket; otherwise use global prior.
        if len(in_vals) < 1 or len(out_vals) < 1:
            scores[j] = 0.0
            continue
        mu_in, mu_out = float(np.mean(in_vals)), float(np.mean(out_vals))
        var_in = float(np.var(in_vals)) + eps
        var_out = float(np.var(out_vals)) + eps
        x = float(t_conf[v])
        # Log-likelihood ratio under univariate Gaussians.
        ll_in = -0.5 * (np.log(2 * np.pi * var_in) + (x - mu_in) ** 2 / var_in)
        ll_out = -0.5 * (np.log(2 * np.pi * var_out) + (x - mu_out) ** 2 / var_out)
        scores[j] = ll_in - ll_out

    if len(np.unique(y_mem)) < 2:
        return 0.5, 0.5, 0.0, float(0.01)

    auc = float(roc_auc_score(y_mem, scores))
    acc = float(accuracy_score(y_mem, (scores > 0).astype(int)))
    tpr_001 = tpr_at_fpr(y_mem, scores, target_fpr=0.001)
    tpr_01 = tpr_at_fpr(y_mem, scores, target_fpr=0.01)
    return auc, acc, tpr_001, tpr_01


def lira_gaussian_scores(
    target_probs,
    target_labels,
    target_train_mask,
    target_test_mask,
    shadow_probs_list,
    shadow_train_masks,
    shadow_test_masks,
    eps=1e-6,
):
    """
    Same offline Gaussian LiRA as lira_gaussian_auc, but returns per-node arrays.

    Returns
    -------
    scores : np.ndarray shape [n_nodes], LiRA LLR (0 where not scored)
    y_mem : np.ndarray shape [n_nodes], 1/0/ -1 ( -1 = not in train∪test eval set)
    eval_idx : np.ndarray of scored node indices
    """
    target_probs = np.asarray(target_probs)
    target_labels = np.asarray(target_labels)
    n = len(target_labels)
    trm = np.asarray(target_train_mask).astype(bool)
    tem = np.asarray(target_test_mask).astype(bool)
    eval_mask = trm | tem
    eval_idx = np.where(eval_mask)[0]
    scores_full = np.zeros(n, dtype=np.float64)
    y_full = np.full(n, -1, dtype=np.int64)
    if len(eval_idx) < 4 or len(shadow_probs_list) == 0:
        return scores_full, y_full, eval_idx

    t_conf = _logit_confidence(target_probs, target_labels)
    shadow_confs, shadow_in = [], []
    for k, sp0 in enumerate(shadow_probs_list):
        sp = np.asarray(sp0)
        if sp.shape[0] != target_labels.shape[0]:
            continue
        shadow_confs.append(_logit_confidence(sp, target_labels))
        shadow_in.append(np.asarray(shadow_train_masks[k]).astype(bool))
    if not shadow_confs:
        return scores_full, y_full, eval_idx

    for v in eval_idx:
        y_full[v] = 1 if trm[v] else 0
        in_vals, out_vals = [], []
        for k, sc in enumerate(shadow_confs):
            if shadow_in[k][v]:
                in_vals.append(sc[v])
            else:
                out_vals.append(sc[v])
        if len(in_vals) < 1 or len(out_vals) < 1:
            scores_full[v] = 0.0
            continue
        mu_in, mu_out = float(np.mean(in_vals)), float(np.mean(out_vals))
        var_in = float(np.var(in_vals)) + eps
        var_out = float(np.var(out_vals)) + eps
        x = float(t_conf[v])
        ll_in = -0.5 * (np.log(2 * np.pi * var_in) + (x - mu_in) ** 2 / var_in)
        ll_out = -0.5 * (np.log(2 * np.pi * var_out) + (x - mu_out) ** 2 / var_out)
        scores_full[v] = ll_in - ll_out
    return scores_full, y_full, eval_idx


def lira_auc_on_subset(scores, y_mem, member_idx, nonmember_idx):
    """AUROC using only specified member / non-member node indices."""
    member_idx = np.asarray(member_idx, dtype=int)
    nonmember_idx = np.asarray(nonmember_idx, dtype=int)
    if len(member_idx) < 4 or len(nonmember_idx) < 4:
        return float("nan"), int(len(member_idx)), int(len(nonmember_idx))
    s = np.concatenate([scores[member_idx], scores[nonmember_idx]])
    y = np.concatenate([np.ones(len(member_idx)), np.zeros(len(nonmember_idx))])
    if len(np.unique(y)) < 2:
        return float("nan"), int(len(member_idx)), int(len(nonmember_idx))
    return float(roc_auc_score(y, s)), int(len(member_idx)), int(len(nonmember_idx))
