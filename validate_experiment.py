import numpy as np

def validate_run(data, p, yn, trm, tem, metrics, nc):
    """
    Fatal conditions raise an AssertionError.
    Warnings return a list of warning strings.
    """
    warnings = []
    
    # 1. Fatal: disjoint masks
    assert not np.any(trm & tem), "FATAL: Train and test masks overlap!"
    
    # 2. Fatal: NaNs
    allowed_nans = {
        'normal_loss_ep1', 'epsd_kl_loss_ep1', 'total_loss_ep1',
        'normal_loss_final', 'epsd_kl_loss_final', 'total_loss_final',
        'actual_dp_epsilon', 'dp_delta', 'dp_clip_norm', 'dp_noise_multiplier',
        'label_only_attack_auc', 'label_only_attack_acc', 'label_only_attack_tpr01',
        'label_only_attack_tpr05', 'label_only_attack_adv', 'label_only_flip_rate',
        'train_ego_gap', 'train_ego_gap_std', 'test_ego_gap', 'test_ego_gap_std', 'ego_gap_diff',
        'shadow_attack_auc', 'shadow_attack_acc', 'shadow_attack_tpr01', 'shadow_attack_tpr05', 'shadow_attack_adv',
        'threshold_attack_auc', 'threshold_attack_acc', 'threshold_attack_tpr01', 'threshold_attack_tpr05', 'threshold_attack_adv'
    }
    
    for k, v in metrics.items():
        if isinstance(v, float) and np.isnan(v):
            if k not in allowed_nans:
                assert False, f"FATAL: Metric {k} is NaN!"
    
    # 3. Fatal: Missing hashes
    assert metrics.get("prediction_sha256") and metrics.get("prediction_sha256") != "N/A", "FATAL: Missing prediction hash"
    # model_sha256 is N/A for MLP/LogReg
    
    if hasattr(data, 'stats') and data.stats:
        assert "graph_hash" in data.stats, "FATAL: Missing graph hash"
        assert "realized_homophily" in data.stats, "FATAL: Missing realized homophily"
        assert "realized_density" in data.stats, "FATAL: Missing realized density"
        
        req_homo = data.stats.get("req_homophily")
        real_homo = data.stats.get("realized_homophily")
        if req_homo and abs(req_homo - real_homo) > 0.1:
            warnings.append(f"WARNING: Realized homophily {real_homo:.3f} deviates from target {req_homo:.3f}")
            
    # Warning: saturated metrics
    ta = metrics.get("test_accuracy", 0.0)
    tf = metrics.get("test_f1", 0.0)
    if ta >= 0.99 or tf >= 0.99:
        warnings.append(f"WARNING: Accuracy {ta:.4f} or F1 {tf:.4f} is critically saturated (>= 0.99).")
        
    # Warning: duplicate predictions
    p_rounded = np.round(p, decimals=4)
    unique_p = np.unique(p_rounded, axis=0)
    if len(unique_p) < 0.05 * len(p):
        warnings.append(f"WARNING: Very few unique posterior vectors: {len(unique_p)} unique out of {len(p)} nodes.")
        
    # Warning: Missing classes in splits
    train_classes = np.unique(yn[trm])
    test_classes = np.unique(yn[tem])
    if len(train_classes) < nc:
        warnings.append(f"WARNING: Train split missing classes. Found {len(train_classes)}, expected {nc}")
    if len(test_classes) < nc:
        warnings.append(f"WARNING: Test split missing classes. Found {len(test_classes)}, expected {nc}")
        
    return warnings

def print_warnings(warnings):
    for w in warnings:
        print(w)
