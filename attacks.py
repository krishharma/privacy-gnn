"""
Membership inference attacks and calibration error.
- Confidence-based attack: train a classifier on confidence/entropy features.
- Threshold attack: simple threshold on true-label confidence.
- Shadow-model attack: train attacker on shadow model, evaluate on target model.
- Expected Calibration Error (ECE).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


def extract_features(probs, labels):
    """
    Extract 4-d feature per sample: max prob, true-label prob, entropy, confidence margin.
    Used by both confidence-based and shadow-model attacks.
    """
    n = len(probs)
    f = np.zeros((n, 4))
    f[:, 0] = probs.max(axis=1)
    for i in range(n):
        f[i, 1] = probs[i, labels[i]] if labels[i] < probs.shape[1] else 0.0
    f[:, 2] = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    f[:, 3] = -(1 - f[:, 0]) * np.log(f[:, 0] + 1e-10)
    return f


def confidence_attack(member_probs, nonmember_probs, member_labels, nonmember_labels, random_state=42):
    """
    Confidence-based MIA: train LR on (member, nonmember) features, report AUC and accuracy.
    Also returns threshold attack (median threshold on true-label confidence).
    Returns: (conf_auc, conf_acc, thresh_auc, thresh_acc).
    """
    fm = extract_features(member_probs, member_labels)
    fn = extract_features(nonmember_probs, nonmember_labels)
    X = np.vstack([fm, fn])
    y = np.concatenate([np.ones(len(fm)), np.zeros(len(fn))])

    rng = np.random.RandomState(int(random_state))
    idx = rng.permutation(len(y))
    mid = len(y) // 2
    clf = LogisticRegression(max_iter=300, random_state=int(random_state))
    clf.fit(X[idx[:mid]], y[idx[:mid]])
    yp = clf.predict_proba(X[idx[mid:]])[:, 1]
    yt = y[idx[mid:]]
    conf_auc = roc_auc_score(yt, yp)
    conf_acc = accuracy_score(yt, (yp > 0.5).astype(int))

    all_conf = np.concatenate([fm[:, 1], fn[:, 1]])
    thresh_auc = roc_auc_score(y, all_conf)
    thresh_acc = accuracy_score(y, (all_conf > np.median(all_conf)).astype(int))

    return conf_auc, conf_acc, thresh_auc, thresh_acc


def shadow_attack(shadow_member_p, shadow_nonmember_p, shadow_member_y, shadow_nonmember_y,
                  target_member_p, target_nonmember_p, target_member_y, target_nonmember_y,
                  random_state=42):
    """
    Shadow-model MIA: train attack classifier on shadow model outputs,
    evaluate on target model outputs. Returns (auc, accuracy).
    """
    fm_s = extract_features(shadow_member_p, shadow_member_y)
    fn_s = extract_features(shadow_nonmember_p, shadow_nonmember_y)
    X_tr = np.vstack([fm_s, fn_s])
    y_tr = np.concatenate([np.ones(len(fm_s)), np.zeros(len(fn_s))])

    fm_t = extract_features(target_member_p, target_member_y)
    fn_t = extract_features(target_nonmember_p, target_nonmember_y)
    X_te = np.vstack([fm_t, fn_t])
    y_te = np.concatenate([np.ones(len(fm_t)), np.zeros(len(fn_t))])

    rng = np.random.RandomState(int(random_state))
    idx = rng.permutation(len(y_tr))
    clf = LogisticRegression(max_iter=300, random_state=int(random_state))
    clf.fit(X_tr[idx], y_tr[idx])
    yp = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, yp)
    acc = accuracy_score(y_te, (yp > 0.5).astype(int))
    return auc, acc


def calibration_error(probs, labels, n_bins=10):
    """
    Expected Calibration Error (ECE) using max-confidence bins.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if probs.size == 0:
        return 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        conf_bin = conf[mask].mean()
        acc_bin = correct[mask].mean()
        weight = mask.mean()
        ece += weight * abs(acc_bin - conf_bin)
    return float(ece)
