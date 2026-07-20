import re

with open("run_final.py", "r") as f:
    code = f.read()

# 1. Update agg_kw to include new metrics
old_agg = """    agg_kw = dict(
        test_acc_mean=("test_accuracy", "mean"),
        test_acc_std=("test_accuracy", "std"),
        attack_auc_mean=("conf_attack_auc", "mean"),
        attack_auc_std=("conf_attack_auc", "std"),
        thresh_auc_mean=("threshold_attack_auc", "mean"),
        thresh_auc_std=("threshold_attack_auc", "std"),
        shadow_auc_mean=("shadow_attack_auc", "mean"),
        shadow_auc_std=("shadow_attack_auc", "std"),
        ece_test_mean=("ece_test", "mean"),
        ece_test_std=("ece_test", "std"),
        label_only_auc_mean=("label_only_attack_auc", "mean"),
        label_only_auc_std=("label_only_attack_auc", "std"),
    )"""

new_agg = """    agg_kw = dict(
        test_acc_mean=("test_accuracy", "mean"),
        test_acc_std=("test_accuracy", "std"),
        test_f1_mean=("test_f1", "mean"),
        test_f1_std=("test_f1", "std"),
        
        attack_auc_mean=("conf_attack_auc", "mean"),
        attack_auc_std=("conf_attack_auc", "std"),
        attack_tpr01_mean=("conf_attack_tpr01", "mean"),
        attack_tpr05_mean=("conf_attack_tpr05", "mean"),
        
        thresh_auc_mean=("threshold_attack_auc", "mean"),
        thresh_auc_std=("threshold_attack_auc", "std"),
        
        shadow_auc_mean=("shadow_attack_auc", "mean"),
        shadow_auc_std=("shadow_attack_auc", "std"),
        shadow_tpr01_mean=("shadow_attack_tpr01", "mean"),
        shadow_tpr05_mean=("shadow_attack_tpr05", "mean"),
        
        ece_test_mean=("ece_test", "mean"),
        ece_test_std=("ece_test", "std"),
        
        label_only_auc_mean=("label_only_attack_auc", "mean"),
        label_only_auc_std=("label_only_attack_auc", "std"),
        label_only_tpr01_mean=("label_only_attack_tpr01", "mean"),
        label_only_tpr05_mean=("label_only_attack_tpr05", "mean"),
        
        peak_memory_mb_mean=("peak_memory_mb", "mean"),
    )"""
code = code.replace(old_agg, new_agg)

# 2. Implement epsd_tuned rule BEFORE calling run_one
old_run = """        try:
            row = run_one(
                ds,
                model,
                dn,
                dp,
                seed,
                data_dir=data_dir,
                device=device,
                training_kwargs=training_kwargs,
                config=config,
            )"""

new_run = """        # epsd_tuned hyperparameter discipline
        if dn == "epsd_tuned":
            dn = "epsd"  # use standard epsd implementation
            if "synthetic_low_" in ds:
                dp = {"lambda_epsd": 5.0}
            else:
                dp = {"lambda_epsd": 1.0}
                
        try:
            row = run_one(
                ds,
                model,
                dn,
                dp,
                seed,
                data_dir=data_dir,
                device=device,
                training_kwargs=training_kwargs,
                config=config,
            )
            # Revert defense name for logging so we can distinguish it in the CSV
            if row.get("defense") == "epsd":
                row["defense"] = "epsd_tuned"
"""
code = code.replace(old_run, new_run)

with open("run_final.py", "w") as f:
    f.write(code)

print("Patched run_final.py")
