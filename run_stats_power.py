"""
Compute bootstrap ΔAUROC CIs and a power-analysis paragraph artifact.
Reads results/core_results.csv (or all_results.csv).
Writes results/summary_delta_bootstrap.csv and results/power_analysis.json.
"""
from __future__ import annotations

import json
import os
import pandas as pd

from config import load_config, ensure_dirs
from stats_utils import (
    run_bootstrap_delta_summary,
    bootstrap_delta_ci,
    power_analysis_paired,
    run_bootstrap_summary,
    run_confirmatory_tests,
)


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    results_dir = cfg["results_dir"]
    path = os.path.join(results_dir, "core_results.csv")
    if not os.path.exists(path):
        path = os.path.join(results_dir, "all_results.csv")
    df = pd.read_csv(path)

    bdf = run_bootstrap_summary(df, output_path=os.path.join(results_dir, "summary_bootstrap.csv"))
    ddf = run_bootstrap_delta_summary(
        df,
        output_path=os.path.join(results_dir, "summary_delta_bootstrap.csv"),
        defenses=("sami", "lbp", "gtd", "label_smoothing", "dropedge", "maskarmor"),
        metrics=("lira_attack_auc", "conf_attack_auc", "lira_tpr_at_0.01_fpr"),
    )
    run_confirmatory_tests(
        df, output_path=os.path.join(results_dir, "significance_confirmatory.csv")
    )

    # Power analysis on headline Cora GraphSAGE conf Δ
    sub = bootstrap_delta_ci(
        df[(df.dataset == "Cora") & (df.model == "GraphSAGE")],
        value_col="conf_attack_auc",
        defense="sami",
    )
    power = {}
    if not sub.empty:
        row = sub.iloc[0]
        power["cora_graphsage_conf"] = power_analysis_paired(
            row["delta_mean"], row["delta_std"], alpha=0.05, power=0.8
        )
        power["cora_graphsage_conf"]["observed_n_seeds"] = int(row["n_seeds"])
        power["cora_graphsage_conf"]["delta_ci"] = [float(row["ci_low"]), float(row["ci_high"])]
    # LiRA as well
    sub2 = bootstrap_delta_ci(
        df[(df.dataset == "Cora") & (df.model == "GraphSAGE")],
        value_col="lira_attack_auc",
        defense="sami",
    )
    if not sub2.empty:
        row = sub2.iloc[0]
        power["cora_graphsage_lira"] = power_analysis_paired(
            row["delta_mean"], row["delta_std"], alpha=0.05, power=0.8
        )
        power["cora_graphsage_lira"]["observed_n_seeds"] = int(row["n_seeds"])
        power["cora_graphsage_lira"]["delta_ci"] = [float(row["ci_low"]), float(row["ci_high"])]

    # Draft paragraph for the paper
    para = (
        "Power analysis (paired normal approximation, two-sided α=0.05, power=0.8). "
        "On Cora GraphSAGE, the observed confidence-attack ΔAUROC "
        f"(SAMI − none) has mean {power.get('cora_graphsage_conf', {}).get('delta_mean', float('nan')):.3f} "
        f"with paired SD {power.get('cora_graphsage_conf', {}).get('delta_std', float('nan')):.3f}, "
        f"implying approximately {power.get('cora_graphsage_conf', {}).get('n_seeds_required_ceil', '?')} "
        f"seeds to detect this effect; we report {power.get('cora_graphsage_conf', {}).get('observed_n_seeds', 5)} "
        "seeds with bootstrap CIs on the paired Δ itself."
    )
    power["paper_paragraph"] = para
    with open(os.path.join(results_dir, "power_analysis.json"), "w") as f:
        json.dump(power, f, indent=2)
    print(para)
    print(f"delta rows: {len(ddf)}; bootstrap rows: {len(bdf)}")


if __name__ == "__main__":
    main()
