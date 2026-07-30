"""
PrivacyGNN main runner: config-driven experiments, results, summary, and significance tests.

Usage:
  python run_final.py

Experiment grid is read from experiment_config.yaml (see config.py). Outputs:
  - results/all_results.csv
  - results/summary.csv
  - results/significance.csv (exploratory paired t-tests, conf_attack_auc vs none)
  - results/significance_lira.csv (exploratory t-tests on lira_attack_auc, if present)
  - results/summary_bootstrap.csv (bootstrap CIs over seeds; not p-values)
"""
import time
import warnings

import pandas as pd
import torch

from config import load_config, get_experiment_list, ensure_dirs
from experiment import run_one
from stats_utils import (
    run_significance_tests,
    run_significance_tests_lira,
    run_bootstrap_summary,
    run_confirmatory_tests,
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
    for i, (ds, model, dn, dp, seed) in enumerate(exps):
        if i % 20 == 0:
            print(
                f"  [{i}/{len(exps)}] {ds}/{model}/{dn} seed={seed} ({time.time() - t0:.0f}s)",
                flush=True,
            )
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
            results.append(row)
            if len(results) % 50 == 0:
                pd.DataFrame(results).to_csv(
                    f"{results_dir}/all_results.partial.csv", index=False
                )
        except Exception as e:
            print(f"  ERR {ds}/{model}/{dn} seed={seed}: {e}", flush=True)

    df = pd.DataFrame(results)
    all_path = f"{results_dir}/all_results.csv"
    df.to_csv(all_path, index=False)
    print(f"\nDone: {len(results)} results in {time.time() - t0:.0f}s")
    print(f"Saved: {all_path}")

    agg_kw = dict(
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
    print("\n" + summary.to_string())

    if len(results) > 0:
        adj = (config.get("stats") or {}).get("multiple_comparison", "holm")
        sig_path = f"{results_dir}/significance.csv"
        sig_df = run_significance_tests(df, output_path=sig_path, adjustment=adj)
        print(f"\nSignificance (paired t-test + {adj}, conf AUC vs none): {sig_path}")
        if not sig_df.empty:
            print(sig_df.head(10).to_string())

        sig_lira_path = f"{results_dir}/significance_lira.csv"
        sig_l = run_significance_tests_lira(df, output_path=sig_lira_path, adjustment=adj)
        if not sig_l.empty:
            print(f"\nSignificance (paired t-test + {adj}, LiRA AUC vs none): {sig_lira_path}")

        conf_path = f"{results_dir}/significance_confirmatory.csv"
        conf_df = run_confirmatory_tests(df, output_path=conf_path, adjustment=adj)
        if not conf_df.empty:
            print(f"\nConfirmatory comparisons (adjusted): {conf_path}")

        boot_path = f"{results_dir}/summary_bootstrap.csv"
        boot_df = run_bootstrap_summary(
            df,
            output_path=boot_path,
            bootstrap_cfg=config.get("bootstrap", {}),
        )
        print(f"\nBootstrap CIs over seeds (not p-values): {boot_path}")
        if not boot_df.empty:
            print(boot_df.head(12).to_string())


if __name__ == "__main__":
    main()
