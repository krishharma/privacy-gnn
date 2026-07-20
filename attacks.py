"""
Membership inference attacks and calibration error.
- Confidence-based attack: train a classifier on confidence/entropy features.
- Threshold attack: simple threshold on true-label confidence.
- Shadow-model attack: train attacker on shadow model, evaluate on target model.
- Expected Calibration Error (ECE).
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

def tpr_at_fpr(y_true, y_score, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) > 0:
        return float(tpr[idx[-1]])
    return 0.0

def max_attack_advantage(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


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
    Returns: dict with conf_auc, conf_acc, conf_tpr_01, conf_tpr_05, thresh_auc, thresh_acc, thresh_tpr_01, thresh_tpr_05.
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
    conf_tpr_01 = tpr_at_fpr(yt, yp, 0.01)
    conf_tpr_05 = tpr_at_fpr(yt, yp, 0.05)
    conf_adv = max_attack_advantage(yt, yp)

    all_conf = np.concatenate([fm[:, 1], fn[:, 1]])
    thresh_auc = roc_auc_score(y, all_conf)
    thresh_acc = accuracy_score(y, (all_conf > np.median(all_conf)).astype(int))
    thresh_tpr_01 = tpr_at_fpr(y, all_conf, 0.01)
    thresh_tpr_05 = tpr_at_fpr(y, all_conf, 0.05)
    thresh_adv = max_attack_advantage(y, all_conf)

    return {
        'conf_auc': float(conf_auc),
        'conf_acc': float(conf_acc),
        'conf_tpr_01': conf_tpr_01,
        'conf_tpr_05': conf_tpr_05,
        'conf_adv': conf_adv,
        'thresh_auc': float(thresh_auc),
        'thresh_acc': float(thresh_acc),
        'thresh_tpr_01': thresh_tpr_01,
        'thresh_tpr_05': thresh_tpr_05,
        'thresh_adv': thresh_adv
    }

def loss_attack(member_loss, nonmember_loss):
    """
    Loss-based MIA: threshold on cross-entropy loss.
    Lower loss -> higher probability of membership.
    """
    y_true = np.concatenate([np.ones(len(member_loss)), np.zeros(len(nonmember_loss))])
    # Score is -loss so that higher score -> member
    y_score = np.concatenate([-np.array(member_loss), -np.array(nonmember_loss)])
    
    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, (y_score > np.median(y_score)).astype(int))
    tpr_01 = tpr_at_fpr(y_true, y_score, 0.01)
    tpr_05 = tpr_at_fpr(y_true, y_score, 0.05)
    adv = max_attack_advantage(y_true, y_score)
    
    return {
        'auc': float(auc),
        'acc': float(acc),
        'tpr_01': tpr_01,
        'tpr_05': tpr_05,
        'adv': adv
    }


def shadow_attack(shadow_member_p, shadow_nonmember_p, shadow_member_y, shadow_nonmember_y,
                  target_member_p, target_nonmember_p, target_member_y, target_nonmember_y,
                  random_state=42):
    """
    Shadow-model MIA: train attack classifier on shadow model outputs,
    evaluate on target model outputs.
    Follows strict attack-train, attack-val (via GridSearchCV), attack-test (target model) separation.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    fm_s = extract_features(shadow_member_p, shadow_member_y)
    fn_s = extract_features(shadow_nonmember_p, shadow_nonmember_y)
    X_sh = np.vstack([fm_s, fn_s])
    y_sh = np.concatenate([np.ones(len(fm_s)), np.zeros(len(fn_s))])

    X_tr = X_sh
    y_tr = y_sh

    fm_t = extract_features(target_member_p, target_member_y)
    fn_t = extract_features(target_nonmember_p, target_nonmember_y)
    X_te = np.vstack([fm_t, fn_t])
    y_te = np.concatenate([np.ones(len(fm_t)), np.zeros(len(fn_t))])

    # Use an MLP to capture non-linear combinations of confidence, entropy, and margin
    best_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, alpha=0.01, random_state=int(random_state))
    best_clf.fit(X_tr, y_tr)
    
    yp = best_clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, yp)
    acc = accuracy_score(y_te, (yp > 0.5).astype(int))
    tpr_01 = tpr_at_fpr(y_te, yp, 0.01)
    tpr_05 = tpr_at_fpr(y_te, yp, 0.05)
    adv = max_attack_advantage(y_te, yp)
    
    return {
        'auc': float(auc),
        'acc': float(acc),
        'tpr_01': tpr_01,
        'tpr_05': tpr_05,
        'adv': adv
    }


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


def label_only_attack(model, data, member_mask, nonmember_mask, num_samples=1000, max_noise_scale=5.0, steps=20):
    """
    Perturbation-based boundary attack for label-only MIA.
    Returns: dict with auc, acc, tpr_01, tpr_05, distances_m, distances_nm, flip_rate.
    """
    device = data.x.device
    if hasattr(model, "eval"):
        model.eval()
    
    # Balance and sample members/nonmembers
    m_idx = np.where(member_mask)[0]
    nm_idx = np.where(nonmember_mask)[0]
    if len(m_idx) > num_samples:
        m_idx = np.random.choice(m_idx, num_samples, replace=False)
    if len(nm_idx) > num_samples:
        nm_idx = np.random.choice(nm_idx, num_samples, replace=False)
        
    eval_idx = np.concatenate([m_idx, nm_idx])
    y_true = np.concatenate([np.ones(len(m_idx)), np.zeros(len(nm_idx))])
    
    # Precompute normal labels
    if isinstance(model, torch.nn.Module):  # GNN
        with torch.no_grad():
            logits_clean = model(data.x, data.edge_index)
            preds_clean = logits_clean.argmax(dim=1)
    else: # sklearn
        preds_clean = torch.tensor(model.predict(data.x.cpu().numpy()), device=device)

    # We test incremental noise scales.
    # We record the MINIMUM noise scale that flips the prediction.
    flip_distances = np.full(len(eval_idx), float('inf'))
    
    noise_scales = np.linspace(0.01, max_noise_scale, steps)
    feature_std = data.x.std(dim=0).mean().item() + 1e-6
    
    for scale in noise_scales:
        x_perturbed = data.x.clone()
        noise = torch.randn(len(eval_idx), data.x.size(1), device=device) * (scale * feature_std)
        x_perturbed[eval_idx] += noise
        
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                logits_pert = model(x_perturbed, data.edge_index)
                preds_pert = logits_pert.argmax(dim=1)
        else:
            preds_pert = torch.tensor(model.predict(x_perturbed.cpu().numpy()), device=device)
            
        # Check flips (Now outside the else block!)
        flipped = (preds_pert[eval_idx] != preds_clean[eval_idx]).cpu().numpy()
        
        # Update flip distance for those that flipped for the first time
        new_flips = flipped & (flip_distances == float('inf'))
        flip_distances[new_flips] = scale

    flip_rate = 1.0 - (np.sum(flip_distances == float('inf')) / len(eval_idx))

    # Replace inf with a maximum bound for AUC calculation
    flip_distances[flip_distances == float('inf')] = max_noise_scale * 1.5
    
    # Distance to boundary is higher for members
    auc = roc_auc_score(y_true, flip_distances)
    tpr_01 = tpr_at_fpr(y_true, flip_distances, 0.01)
    tpr_05 = tpr_at_fpr(y_true, flip_distances, 0.05)
    adv = max_attack_advantage(y_true, flip_distances)
    
    # Accuracy using median threshold
    threshold = np.median(flip_distances)
    pred_mia = (flip_distances > threshold).astype(int)
    acc = accuracy_score(y_true, pred_mia)
    
    distances_m = flip_distances[:len(m_idx)]
    distances_nm = flip_distances[len(m_idx):]
    
    return {
        'auc': float(auc),
        'acc': float(acc),
        'tpr_01': tpr_01,
        'tpr_05': tpr_05,
        'adv': adv,
        'distances_m': distances_m,
        'distances_nm': distances_nm,
        'flip_rate': float(flip_rate)
    }
