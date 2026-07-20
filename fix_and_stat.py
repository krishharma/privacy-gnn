import pandas as pd
import json

df = pd.read_csv('results/all_results.csv')

# Fix duplicates: epsd_tuned and epsd_ablation were both named epsd_tuned
mask = df['defense_params'].astype(str).str.contains('ablation')
df.loc[mask, 'defense'] = 'epsd_ablation'
df.to_csv('results/all_results.csv', index=False)
print("Fixed duplicate labels in all_results.csv.")

# Load the logic from run_final.py manually
from stats_utils import (
    run_significance_tests,
    run_significance_tests_lira,
    run_bootstrap_summary,
)
import config
import run_final

cfg = config.load_config()
results_dir = cfg["results_dir"]

agg_kw = dict(
    test_acc_mean=("test_accuracy", "mean"),
    test_acc_std=("test_accuracy", "std"),
    test_f1_mean=("test_f1", "mean"),
    test_f1_std=("test_f1", "std"),
    
    attack_auc_mean=("conf_attack_auc", "mean"),
    attack_auc_std=("conf_attack_auc", "std"),
    attack_tpr01_mean=("conf_attack_tpr01", "mean"),
    attack_tpr05_mean=("conf_attack_tpr05", "mean"),
    attack_adv_mean=("conf_attack_adv", "mean"),
    
    thresh_auc_mean=("threshold_attack_auc", "mean"),
    thresh_auc_std=("threshold_attack_auc", "std"),
    
    shadow_auc_mean=("shadow_attack_auc", "mean"),
    shadow_auc_std=("shadow_attack_auc", "std"),
    shadow_tpr01_mean=("shadow_attack_tpr01", "mean"),
    shadow_tpr05_mean=("shadow_attack_tpr05", "mean"),
    shadow_adv_mean=("shadow_attack_adv", "mean"),
    
    ece_test_mean=("ece_test", "mean"),
    ece_test_std=("ece_test", "std"),
    
    label_only_auc_mean=("label_only_attack_auc", "mean"),
    label_only_auc_std=("label_only_attack_auc", "std"),
    label_only_tpr01_mean=("label_only_attack_tpr01", "mean"),
    label_only_tpr05_mean=("label_only_attack_tpr05", "mean"),
    label_only_adv_mean=("label_only_attack_adv", "mean"),
    
    peak_memory_mb_mean=("peak_memory_mb", "mean"),
    
    train_ego_gap_mean=("train_ego_gap", "mean"),
    test_ego_gap_mean=("test_ego_gap", "mean"),
    ego_gap_diff_mean=("ego_gap_diff", "mean"),
)

if "lira_attack_auc" in df.columns:
    agg_kw["lira_auc_mean"] = ("lira_attack_auc", "mean")
    agg_kw["lira_auc_std"] = ("lira_attack_auc", "std")
if "dp_epsilon" in df.columns:
    agg_kw["dp_epsilon_mean"] = ("dp_epsilon", "mean")

summary = df.groupby(["dataset", "model", "defense"]).agg(**agg_kw).round(4)
summary_path = f"{results_dir}/summary.csv"
summary.to_csv(summary_path)
print(f"Saved: {summary_path}")

sig_path = f"{results_dir}/significance.csv"
sig_df = run_significance_tests(df, output_path=sig_path)
print(f"\nExploratory significance (paired t-test, conf AUC vs none): {sig_path}")

boot_path = f"{results_dir}/summary_bootstrap.csv"
boot_df = run_bootstrap_summary(
    df,
    output_path=boot_path,
    bootstrap_cfg=cfg.get("bootstrap", {}),
)
print(f"\nBootstrap CIs over seeds (not p-values): {boot_path}")

try:
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    df_gnn = df[df["model"].isin(["GCN", "GraphSAGE"])].dropna(subset=["ego_gap_diff", "conf_attack_auc"])
    if not df_gnn.empty:
        pearson_r, pearson_p = stats.pearsonr(df_gnn["ego_gap_diff"], df_gnn["conf_attack_auc"])
        spearman_r, spearman_p = stats.spearmanr(df_gnn["ego_gap_diff"], df_gnn["conf_attack_auc"])
        print(f"\nEgo-gap Difference vs Conf MIA AUROC Correlation:")
        print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.4e})")
        print(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.4e})")
        
        plt.figure(figsize=(6, 4))
        for model_name in ["GCN", "GraphSAGE"]:
            subset = df_gnn[df_gnn["model"] == model_name]
            plt.scatter(subset["ego_gap_diff"], subset["conf_attack_auc"], label=model_name, alpha=0.6)
        plt.xlabel("Ego-Gap Difference (Train - Held-out)")
        plt.ylabel("Confidence Attack AUROC")
        plt.title("MIA Risk vs Ego-Gap Difference")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        scatter_path = f"{results_dir}/ego_gap_scatter.png"
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print(f"Ego-gap scatterplot saved to: {scatter_path}")
except Exception as e:
    print(f"Failed to generate ego-gap correlation/plot: {e}")
