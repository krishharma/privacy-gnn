import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import load_config
from experiment import run_one

def main():
    print("Starting Phase 2: EPSD Hyperparameter Search")
    config = load_config("experiment_config_paper.yaml")
    
    datasets = ["Cora", "Citeseer"]
    models = ["GCN", "GraphSAGE"]
    seed = 42
    lambdas = [0.1, 0.5, 1.0, 5.0]
    
    results = []
    
    # Run the hyperparameter search
    for ds in datasets:
        for m in models:
            # Get baseline
            print(f"Running {ds} - {m} - none")
            res_none = run_one(ds, m, "none", {}, seed, config=config)
            res_none["lambda_epsd"] = 0.0
            results.append(res_none)
            
            for l in lambdas:
                print(f"Running {ds} - {m} - epsd (lambda={l})")
                res = run_one(ds, m, "epsd", {"lambda_epsd": l}, seed, config=config)
                res["lambda_epsd"] = l
                results.append(res)
                
    df = pd.DataFrame(results)
    df.to_csv("epsd_lambda_search.csv", index=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # We want to plot Target Accuracy vs Confidence Attack AUROC
    for ds in datasets:
        for m in models:
            subset = df[(df["dataset"] == ds) & (df["model"] == m)].sort_values("lambda_epsd")
            
            # None defense
            base = subset[subset["lambda_epsd"] == 0.0].iloc[0]
            
            # Epsd defense
            epsd_pts = subset[subset["lambda_epsd"] > 0.0]
            
            plt.plot(epsd_pts["conf_attack_auc"], epsd_pts["test_accuracy"], marker='o', label=f"{ds} {m} EPSD")
            plt.scatter([base["conf_attack_auc"]], [base["test_accuracy"]], marker='*', s=150, label=f"{ds} {m} None")
            
            # Annotate lambdas
            for _, row in epsd_pts.iterrows():
                plt.annotate(f"λ={row['lambda_epsd']}", (row["conf_attack_auc"], row["test_accuracy"]), 
                             textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

    plt.xlabel("Confidence Attack AUROC (Lower is better)")
    plt.ylabel("Target Model Test Accuracy (Higher is better)")
    plt.title("EPSD Defense Trade-off: Accuracy vs AUROC")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("epsd_lambda_search.png")
    plt.close()
    
    # Select best lambda
    # We want a lambda that reduces AUROC significantly while maintaining accuracy.
    # We can define a heuristic: maximize (Acc - base_Acc) - 2 * (AUROC - base_AUROC).
    # Since AUROC reduction is good (negative change), and Acc reduction is bad (negative change).
    best_lambda = None
    best_score = -9999
    
    scores = {l: [] for l in lambdas}
    
    for ds in datasets:
        for m in models:
            subset = df[(df["dataset"] == ds) & (df["model"] == m)]
            base = subset[subset["lambda_epsd"] == 0.0].iloc[0]
            
            for l in lambdas:
                l_pt = subset[subset["lambda_epsd"] == l].iloc[0]
                
                delta_acc = l_pt["test_accuracy"] - base["test_accuracy"]
                delta_auc = l_pt["conf_attack_auc"] - base["conf_attack_auc"]
                
                # Heuristic: 1% accuracy drop is worth roughly 2% AUROC drop
                score = delta_acc - 0.5 * delta_auc
                scores[l].append(score)
                
    avg_scores = {l: np.mean(s) for l, s in scores.items()}
    best_lambda = max(avg_scores, key=avg_scores.get)
    
    print("\n" + "="*80)
    print("PHASE 2 STOP AND REPORT")
    print("="*80)
    print(f"Best lambda found: {best_lambda}")
    print("Appending to experiment_config_paper.yaml...")
    
    # Append to experiment_config_paper.yaml
    with open("experiment_config_paper.yaml", "a") as f:
        f.write(f"  - name: epsd\n    params: {{ lambda_epsd: {float(best_lambda)} }}\n")
        
    print("Done. Ready for Phase 3.")

if __name__ == "__main__":
    main()
