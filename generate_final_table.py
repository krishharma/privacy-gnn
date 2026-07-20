import pandas as pd
import numpy as np
from scipy import stats
from stats_utils import bootstrap_ci_over_seeds

def benjamini_hochberg(p_values):
    p_values = np.array(p_values)
    valid_idx = ~np.isnan(p_values)
    if not valid_idx.any():
        return p_values
    valid_p = p_values[valid_idx]
    
    n = len(valid_p)
    sorted_indices = np.argsort(valid_p)
    sorted_p = valid_p[sorted_indices]
    fdr_p = np.zeros(n)
    
    for i in range(n):
        rank = i + 1
        fdr_p[sorted_indices[i]] = sorted_p[i] * n / rank
        
    for i in range(n - 2, -1, -1):
        fdr_p[sorted_indices[i]] = min(fdr_p[sorted_indices[i]], fdr_p[sorted_indices[i+1]])
        
    fdr_p = np.minimum(fdr_p, 1.0)
    
    out_p = np.full_like(p_values, np.nan)
    out_p[valid_idx] = fdr_p
    return out_p

def compute_p_values(df, metrics):
    # Paired t-test vs 'none' defense
    p_values = {m: [] for m in metrics}
    keys = []
    
    groups = df.groupby(["dataset", "model"])
    
    for (ds, model), group in groups:
        none_rows = group[group["defense"] == "none"]
        if none_rows.empty:
            for defense, defense_params in group[["defense", "defense_params"]].drop_duplicates().values:
                keys.append((ds, model, defense, defense_params))
                for m in metrics:
                    p_values[m].append(np.nan)
            continue
            
        none_vals = none_rows.set_index("seed")
        
        for defense, defense_params in group[["defense", "defense_params"]].drop_duplicates().values:
            keys.append((ds, model, defense, defense_params))
            if defense == "none":
                for m in metrics:
                    p_values[m].append(np.nan)
                continue
                
            def_vals = group[(group["defense"] == defense) & (group["defense_params"] == defense_params)].set_index("seed")
            common = none_vals.index.intersection(def_vals.index)
            
            for m in metrics:
                if len(common) < 2:
                    p_values[m].append(np.nan)
                    continue
                a = none_vals.loc[common, m].values
                b = def_vals.loc[common, m].values
                
                valid = ~(np.isnan(a) | np.isnan(b))
                if valid.sum() < 2:
                    p_values[m].append(np.nan)
                    continue
                    
                _, p = stats.ttest_rel(a[valid], b[valid], alternative="two-sided")
                p_values[m].append(p)
                
    res = pd.DataFrame(keys, columns=["dataset", "model", "defense", "defense_params"])
    for m in metrics:
        res[f"{m}_p_val"] = p_values[m]
        res[f"{m}_fdr_q_val"] = benjamini_hochberg(p_values[m])
        
    return res

def format_ci(mean, lo, hi):
    if np.isnan(mean):
        return "-"
    if np.isnan(lo) or np.isnan(hi):
        return f"{mean:.4f}"
    return f"{mean:.4f} ({lo:.4f}, {hi:.4f})"

def generate_table(df, name):
    if df.empty:
        return
        
    metrics = [
        "test_accuracy", "test_f1", 
        "shadow_attack_auc", "shadow_attack_tpr01", "shadow_attack_tpr05", "shadow_attack_adv",
        "conf_attack_auc", "conf_attack_tpr01",
        "label_only_attack_auc", "label_only_attack_tpr01"
    ]
    
    boot_df = bootstrap_ci_over_seeds(
        df,
        group_cols=["dataset", "model", "defense", "defense_params"],
        metric_cols=metrics
    )
    p_df = compute_p_values(df, metrics)
    
    # Pivot boot_df
    formatted_data = []
    for (ds, model, defense, dp), g in boot_df.groupby(["dataset", "model", "defense", "defense_params"]):
        row = {"dataset": ds, "model": model, "defense": defense, "defense_params": dp, "n_seeds": g["n_seeds"].max()}
        for _, r in g.iterrows():
            m = r["metric"]
            row[m] = format_ci(r["mean"], r["ci_low"], r["ci_high"])
        formatted_data.append(row)
        
    final_df = pd.DataFrame(formatted_data)
    final_df = pd.merge(final_df, p_df, on=["dataset", "model", "defense", "defense_params"], how="left")
    
    out_file = f"results/{name}_table.csv"
    final_df.to_csv(out_file, index=False)
    print(f"Saved {name} table to {out_file}")
    
def generate_pareto_plots(df):
    import matplotlib.pyplot as plt
    import json
    import os
    
    if df.empty:
        return
        
    os.makedirs("figures", exist_ok=True)
    
    # Evaluate privacy risk via Shadow Attack AUC or TPR
    risk_metric = "shadow_attack_auc"
    if risk_metric not in df.columns:
        risk_metric = "conf_attack_auc"
        
    # Group to get means over seeds
    df_mean = df.groupby(["dataset", "model", "defense", "defense_params"]).mean(numeric_only=True).reset_index()
    
    datasets = ["Cora", "Citeseer", "synthetic_high_medium", "synthetic_low_medium"]
    models = ["GCN", "GraphSAGE"]
    
    for ds in datasets:
        for model in models:
            sub = df_mean[(df_mean["dataset"] == ds) & (df_mean["model"] == model)]
            if sub.empty:
                continue
                
            plt.figure(figsize=(8, 6))
            
            # Baseline (none)
            none_sub = sub[sub["defense"] == "none"]
            if not none_sub.empty:
                plt.scatter(none_sub["test_accuracy"], none_sub[risk_metric], color='black', marker='*', s=150, label="Undefended Baseline")
            
            colors = {"epsd": "red", "dropedge": "blue", "label_smoothing": "green", "edge_sparsification": "orange", "dp_sgd": "purple"}
            
            for defense in ["epsd", "dropedge", "label_smoothing", "edge_sparsification", "dp_sgd"]:
                def_sub = sub[sub["defense"] == defense]
                if def_sub.empty:
                    continue
                    
                # Extract the hyperparameter value to sort
                def extract_param(param_str):
                    try:
                        p = json.loads(param_str)
                        if defense == "epsd": return p.get("lambda_epsd", 0)
                        if defense == "dropedge": return p.get("rate", 0)
                        if defense == "edge_sparsification": return p.get("rate", 0)
                        if defense == "label_smoothing": return p.get("alpha", 0)
                        if defense == "dp_sgd": return p.get("noise_multiplier", 0)
                    except:
                        return 0
                    return 0
                    
                def_sub = def_sub.copy()
                def_sub["param_val"] = def_sub["defense_params"].apply(extract_param)
                def_sub = def_sub.sort_values("param_val")
                
                plt.plot(def_sub["test_accuracy"], def_sub[risk_metric], marker='o', color=colors.get(defense, "gray"), label=defense)
                
            plt.xlabel("Utility (Test Accuracy)")
            plt.ylabel(f"Privacy Risk ({risk_metric})")
            plt.title(f"Privacy-Utility Pareto Frontier\n{ds} - {model}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Invert X axis so better utility is to the right
            # Better privacy is down. So bottom-right is optimal.
            
            out_path = f"figures/pareto_{ds}_{model}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved Pareto plot to {out_path}")

def main():
    try:
        df = pd.read_csv("results/results_raw_per_seed.csv")
    except FileNotFoundError:
        print("results/results_raw_per_seed.csv not found. Experiments may still be running.")
        return
        
    df_ogb = df[df["dataset"] == "ogbn-arxiv"]
    df_small = df[df["dataset"] != "ogbn-arxiv"]
    
    generate_table(df_ogb, "ogb_scalability")
    generate_table(df_small, "small_graph_comparison")
    generate_pareto_plots(df)

if __name__ == "__main__":
    main()
