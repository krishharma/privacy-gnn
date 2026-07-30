"""Summarize core_results.csv / all_results.csv into paper-ready tables."""
import os
import pandas as pd
from stats_utils import run_significance_tests, run_significance_tests_lira, run_confirmatory_tests, run_bootstrap_summary

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")


def main():
    path = os.path.join(RES, "core_results.csv")
    if not os.path.isfile(path):
        path = os.path.join(RES, "all_results.csv")
    df = pd.read_csv(path)
    print(f"Loaded {path} ({len(df)} rows)")

    priv = "lira_attack_auc" if "lira_attack_auc" in df.columns else "conf_attack_auc"
    aggs = {
        "acc_mean": ("test_accuracy", "mean"),
        "acc_std": ("test_accuracy", "std"),
        "conf_mean": ("conf_attack_auc", "mean"),
        "conf_std": ("conf_attack_auc", "std"),
        "lira_mean": (priv, "mean"),
        "lira_std": (priv, "std"),
        "ece_mean": ("ece_test", "mean"),
    }
    if "lira_tpr_at_0.01_fpr" in df.columns:
        aggs["lira_tpr01_mean"] = ("lira_tpr_at_0.01_fpr", "mean")
        aggs["lira_tpr001_mean"] = ("lira_tpr_at_0.001_fpr", "mean")
    if "mlp_phi_attack_auc" in df.columns:
        aggs["mlp_phi_mean"] = ("mlp_phi_attack_auc", "mean")
    g = df.groupby(["dataset", "model", "defense"]).agg(**aggs).round(4)
    out = os.path.join(RES, "paper_tables_summary.csv")
    g.to_csv(out)
    print(g.to_string())
    print(f"\nSaved {out}")

    run_significance_tests(df, os.path.join(RES, "significance.csv"))
    run_significance_tests_lira(df, os.path.join(RES, "significance_lira.csv"))
    run_confirmatory_tests(df, os.path.join(RES, "significance_confirmatory.csv"))
    run_bootstrap_summary(df, os.path.join(RES, "summary_bootstrap.csv"))
    from stats_utils import run_bootstrap_delta_summary

    run_bootstrap_delta_summary(df, os.path.join(RES, "summary_delta_bootstrap.csv"))

    # Snapshot
    snap = os.path.join(RES, "paper_release")
    os.makedirs(snap, exist_ok=True)
    df.to_csv(os.path.join(snap, "all_results.csv"), index=False)
    g.to_csv(os.path.join(snap, "paper_tables_summary.csv"))
    print(f"Snapshotted to {snap}")


if __name__ == "__main__":
    main()
