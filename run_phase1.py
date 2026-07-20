import os
import sys
import pandas as pd
import numpy as np
import torch
import scipy.stats as stats
import matplotlib.pyplot as plt

from config import load_config
from experiment import _load_target_data
from models import GCN, SAGE
from training import train_gnn
from epsd_utils import compute_ego_gap

def run_diagnostic():
    print("Starting Phase 1: Ego-Gap Diagnostic")
    # Force use of paper config as per instructions
    config = load_config("experiment_config_paper.yaml")
    
    device = torch.device(config.get("device", "cpu"))
    data_dir = config["data_dir"]
    tk = config.get("training", {})
    
    datasets = [d for d in config["datasets"] if not d.startswith("ogb") and d != "Reddit"]
    models = ["GCN", "GraphSAGE"]
    seeds = config["seeds"]
    
    results = []
    
    # Track the original AUROCs for correlation
    # We can load them from results/summary.csv
    summary_path = os.path.join(config["results_dir"], "summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
    else:
        # Fallback to all_results if summary doesn't exist
        all_res_path = os.path.join(config["results_dir"], "all_results.csv")
        if os.path.exists(all_res_path):
            all_res = pd.read_csv(all_res_path)
            summary_df = all_res.groupby(["dataset", "model", "defense"]).mean(numeric_only=True).reset_index()
        else:
            print("WARNING: Could not find existing results to correlate with!")
            summary_df = pd.DataFrame()

    total_runs = len(datasets) * len(models) * len(seeds)
    run_idx = 0
    
    for ds in datasets:
        for m in models:
            for seed in seeds:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] Retraining {ds} - {m} - seed {seed}...")
                
                # Setup
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                data, num_classes, num_features = _load_target_data(ds, data_dir, seed, True)
                data = data.to(device)
                
                # Train model
                model = (GCN if m == "GCN" else SAGE)(ic=num_features, h=64, oc=num_classes).to(device)
                
                train_gnn(
                    model,
                    data,
                    device,
                    epochs=int(tk.get("epochs", 50)),
                    lr=float(tk.get("lr", 0.01)),
                    weight_decay=float(tk.get("weight_decay", 5e-4)),
                )
                
                # Compute ego gap
                trm = data.train_mask.cpu().numpy()
                tem = data.test_mask.cpu().numpy()
                
                train_indices = np.where(trm)[0]
                test_indices = np.where(tem)[0]
                
                g_train = compute_ego_gap(model, data, train_indices)
                g_test = compute_ego_gap(model, data, test_indices)
                
                results.append({
                    "dataset_id": ds,
                    "model": m,
                    "seed": seed,
                    "mean_ego_gap_train": float(g_train.mean()),
                    "mean_ego_gap_heldout": float(g_test.mean()),
                    "ego_gap_difference": float(g_train.mean() - g_test.mean()),
                    "std_ego_gap_train": float(g_train.std()),
                    "std_ego_gap_heldout": float(g_test.std()),
                })

    res_df = pd.DataFrame(results)
    res_df.to_csv("ego_gap_diagnostic_results.csv", index=False)
    
    # 1.4 Correlate with existing attack AUROC
    # Aggregate across seeds
    agg_df = res_df.groupby(["dataset_id", "model"]).agg({
        "ego_gap_difference": "mean",
        "mean_ego_gap_train": "mean",
        "mean_ego_gap_heldout": "mean"
    }).reset_index()
    
    correlation_data = []
    
    for _, row in agg_df.iterrows():
        ds = row["dataset_id"]
        m = row["model"]
        
        # Look up existing AUROC for defense='none'
        if not summary_df.empty:
            match = summary_df[(summary_df["dataset"] == ds) & (summary_df["model"] == m) & (summary_df["defense"] == "none")]
            if not match.empty:
                auroc = match.iloc[0]["attack_auc_mean"] if "attack_auc_mean" in match.columns else match.iloc[0]["conf_attack_auc"]
            else:
                auroc = np.nan
        else:
            auroc = np.nan
            
        correlation_data.append({
            "dataset_id": ds,
            "model": m,
            "ego_gap_difference": row["ego_gap_difference"],
            "conf_attack_auc": auroc
        })
        
    corr_df = pd.DataFrame(correlation_data).dropna()
    corr_df.to_csv("ego_gap_correlation_data.csv", index=False)
    
    if len(corr_df) > 1:
        pearson_r, pearson_p = stats.pearsonr(corr_df["ego_gap_difference"], corr_df["conf_attack_auc"])
        spearman_r, spearman_p = stats.spearmanr(corr_df["ego_gap_difference"], corr_df["conf_attack_auc"])
        
        plt.figure(figsize=(8, 6))
        for m in models:
            subset = corr_df[corr_df["model"] == m]
            plt.scatter(subset["ego_gap_difference"], subset["conf_attack_auc"], label=m)
        plt.xlabel("Ego-Gap Difference (Train - Heldout)")
        plt.ylabel("Confidence Attack AUROC")
        plt.title(f"Ego-Gap vs Attack AUROC\nPearson r={pearson_r:.3f}, p={pearson_p:.3e}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("ego_gap_vs_attack_auroc.png")
        plt.close()
    else:
        pearson_r, pearson_p, spearman_r, spearman_p = np.nan, np.nan, np.nan, np.nan
        
    print("\n" + "="*80)
    print("PHASE 1 STOP AND REPORT")
    print("="*80)
    print(f"Pearson Correlation: r = {pearson_r:.3f}, p = {pearson_p:.3e}")
    print(f"Spearman Correlation: rho = {spearman_r:.3f}, p = {spearman_p:.3e}")
    
    mean_gap_diff = agg_df["ego_gap_difference"].mean()
    print(f"Average ego-gap difference across all regimes: {mean_gap_diff:.4f}")
    if mean_gap_diff > 0:
        print("-> The ego-gap is LARGER for training nodes than held-out nodes on average.")
    else:
        print("-> The ego-gap is NOT larger for training nodes on average.")
        
    # Check low-homophily sparse GCN
    lh_sparse_gcn = agg_df[(agg_df["dataset_id"] == "synthetic_low_sparse") & (agg_df["model"] == "GCN")]
    if not lh_sparse_gcn.empty:
        val = lh_sparse_gcn.iloc[0]["ego_gap_difference"]
        print(f"Ego-gap difference for GCN on low-homophily sparse: {val:.4f}")
        if val == agg_df["ego_gap_difference"].max():
            print("-> This effect is STRONGEST in the low-homophily sparse regime for GCN.")
        else:
            print("-> This effect is NOT the strongest in the low-homophily sparse regime.")
            
    if abs(pearson_r) < 0.3 or pearson_r < 0:
        print("\nWARNING: Correlation is weak or in the unexpected direction. "
              "This may require revisiting the EPSD approach, but we will proceed to Phase 2 to gather more evidence.")
    else:
        print("\nSUCCESS: Ego-gap strongly predicts membership leakage vulnerability!")

if __name__ == "__main__":
    run_diagnostic()
