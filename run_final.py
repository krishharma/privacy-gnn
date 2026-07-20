"""
PrivacyGNN main runner: config-driven experiments, results, summary, and significance tests.

Usage:
  python run_final.py

Experiment grid is read from experiment_config.yaml (see config.py). Outputs:
  - results/results_raw_per_seed.csv
  - results/epsd_forensic_audit.csv
  - results/results_audit.csv
  - results/summary.csv
  - results/summary_bootstrap.csv (bootstrap CIs over seeds; not p-values)
"""
import time
import warnings
import math
import pandas as pd
import torch

from config import load_config, get_experiment_list, ensure_dirs
from experiment import run_one
from stats_utils import (
    run_significance_tests,
    run_bootstrap_summary,
)

warnings.filterwarnings("ignore")
torch.set_num_threads(2)

def main():
    config = load_config()
    ensure_dirs(config)
    device = torch.device(config.get("device", "cpu"))
    data_dir = config["data_dir"]
    results_dir = config["results_dir"]
    training_kwargs = config.get("training", {})

    exps = get_experiment_list(config)
    cfg_src = config.get("_config_path") or "built-in defaults"
    print(f"Total: {len(exps)} experiments (config: {cfg_src})", flush=True)
    t0 = time.time()
    results = []
    epsd_audit_rows = []
    
    for i, (ds, model, dn, dp, seed) in enumerate(exps):
        if i % 20 == 0:
            print(
                f"  [{i}/{len(exps)}] {ds}/{model}/{dn} seed={seed} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        # epsd_tuned hyperparameter discipline
        if dn == "epsd_tuned":
            dn = "epsd"  # use standard epsd implementation
            if ds == "synthetic_low_sparse":
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
                
                # FORENSIC AUDIT ASSERTIONS
                kl_ep1 = row.get("epsd_kl_loss_ep1")
                assert kl_ep1 is not None and not math.isnan(kl_ep1), f"EPSD run missing kl loss on ep 1 for {ds}/{model}"
                assert kl_ep1 > 1e-8, f"EPSD run KL loss on ep 1 is {kl_ep1}, should be > 1e-8"
                assert row.get("model_sha256") != "N/A", "Model SHA is N/A for EPSD!"
                
                epsd_audit_rows.append({
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "defense": row["defense"],
                    "seed": row["seed"],
                    "lambda_epsd": row["lambda_epsd"],
                    "ablation": row["epsd_ablation"],
                    "kl_loss_ep1": row["epsd_kl_loss_ep1"],
                    "kl_loss_final": row["epsd_kl_loss_final"],
                    "model_sha256": row["model_sha256"],
                    "prediction_sha256": row["prediction_sha256"],
                })

            results.append(row)
            if len(results) % 50 == 0:
                pd.DataFrame(results).to_csv(
                    f"{results_dir}/results_raw_per_seed.csv", index=False
                )
        except Exception as e:
            print(f"  ERR {ds}/{model}/{dn} seed={seed}: {e}", flush=True)

    df = pd.DataFrame(results)
    raw_path = f"{results_dir}/results_raw_per_seed.csv"
    df.to_csv(raw_path, index=False)
    
    # epsd_forensic_audit.csv
    if epsd_audit_rows:
        epsd_audit_df = pd.DataFrame(epsd_audit_rows)
        epsd_audit_path = f"{results_dir}/epsd_forensic_audit.csv"
        epsd_audit_df.to_csv(epsd_audit_path, index=False)
        print(f"Saved: {epsd_audit_path}")
    
    # results_audit.csv (renamed from all_results.csv)
    audit_path = f"{results_dir}/results_audit.csv"
    df.to_csv(audit_path, index=False)
    print(f"\nDone: {len(results)} results in {time.time() - t0:.0f}s")
    print(f"Saved: {raw_path}")
    print(f"Saved: {audit_path}")

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
        
        loss_auc_mean=("loss_attack_auc", "mean"),
        loss_auc_std=("loss_attack_auc", "std"),
        loss_tpr01_mean=("loss_attack_tpr01", "mean"),
        loss_tpr05_mean=("loss_attack_tpr05", "mean"),
        loss_adv_mean=("loss_attack_adv", "mean"),
        
        train_ego_gap_mean=("train_ego_gap", "mean"),
        test_ego_gap_mean=("test_ego_gap", "mean"),
        ego_gap_diff_mean=("ego_gap_diff", "mean"),
    )
    if "actual_dp_epsilon" in df.columns:
        agg_kw["dp_epsilon_mean"] = ("actual_dp_epsilon", "mean")

    summary = df.groupby(["dataset", "model", "defense"]).agg(**agg_kw).round(4)
    summary_path = f"{results_dir}/summary.csv"
    summary.to_csv(summary_path)
    print(f"Saved: {summary_path}")
    print("\n" + summary.to_string())

    if len(results) > 0:
        sig_path = f"{results_dir}/significance.csv"
        sig_df = run_significance_tests(df, output_path=sig_path)
        print(f"\nExploratory significance (paired t-test, conf AUC vs none): {sig_path}")
        if not sig_df.empty:
            print(sig_df.head(10).to_string())

        boot_path = f"{results_dir}/summary_bootstrap.csv"
        boot_df = run_bootstrap_summary(
            df,
            output_path=boot_path,
            bootstrap_cfg=config.get("bootstrap", {}),
        )
        print(f"\nBootstrap CIs over seeds (not p-values): {boot_path}")
        if not boot_df.empty:
            print(boot_df.head(12).to_string())

        # Compute Pearson/Spearman correlation between ego-gap difference and MIA AUROC
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

if __name__ == "__main__":
    main()
