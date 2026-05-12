"""
Statistical significance testing for experiment results.
Used to compare attack AUC / utility across defenses or models (e.g. paired tests across seeds).
"""
import numpy as np
import pandas as pd
from scipy import stats


def paired_ttest(df, group_col, value_col, baseline_group, alternative="two-sided"):
    """
    For each level of group_col, run a paired t-test vs baseline_group (across seeds).
    Assumes each (dataset, model, defense) has one row per seed; we compare value_col.
    Returns a small DataFrame: group, mean_diff, t_stat, p_value.
    """
    groups = df[group_col].unique()
    baseline_vals = df.loc[df[group_col] == baseline_group].set_index(["dataset", "model", "seed"])[value_col]
    results = []
    for g in groups:
        if g == baseline_group:
            continue
        other = df.loc[df[group_col] == g].set_index(["dataset", "model", "seed"])[value_col]
        # Align by (dataset, model, seed)
        common = baseline_vals.index.intersection(other.index)
        if len(common) < 2:
            results.append({"group": g, "mean_diff": np.nan, "t_stat": np.nan, "p_value": np.nan})
            continue
        a = baseline_vals.reindex(common).dropna().values
        b = other.reindex(common).dropna().values
        if len(a) != len(b) or len(a) < 2:
            results.append({"group": g, "mean_diff": np.nan, "t_stat": np.nan, "p_value": np.nan})
            continue
        t, p = stats.ttest_rel(a, b, alternative=alternative)
        results.append({"group": g, "mean_diff": float(np.mean(b - a)), "t_stat": float(t), "p_value": float(p)})
    return pd.DataFrame(results)


def significance_vs_no_defense(df, value_col="conf_attack_auc"):
    """
    For each (dataset, model, defense), test whether value_col differs from the same
    (dataset, model) with defense='none' (paired by seed). Returns DataFrame with
    dataset, model, defense, mean, mean_none, diff, p_value.
    """
    out = []
    for (ds, model), g in df.groupby(["dataset", "model"]):
        none_rows = g[g["defense"] == "none"]
        if none_rows.empty:
            continue
        none_vals = none_rows.set_index("seed")[value_col]
        for defense in g["defense"].unique():
            if defense == "none":
                continue
            def_rows = g[g["defense"] == defense]
            def_vals = def_rows.set_index("seed")[value_col]
            common = none_vals.index.intersection(def_vals.index)
            if len(common) < 2:
                continue
            a = none_vals.reindex(common).dropna().values
            b = def_vals.reindex(common).dropna().values
            if len(a) != len(b):
                continue
            t, p = stats.ttest_rel(a, b, alternative="two-sided")
            out.append({
                "dataset": ds, "model": model, "defense": defense,
                "mean_none": float(np.mean(a)), "mean_defense": float(np.mean(b)),
                "diff": float(np.mean(b - a)), "p_value": float(p),
            })
    return pd.DataFrame(out)


def run_significance_tests(results_df, output_path=None):
    """
    Run pairwise significance tests and optionally save.
    - Compare each defense vs no defense (paired by seed) for conf_attack_auc.
    Returns the significance DataFrame.
    """
    sig_df = significance_vs_no_defense(results_df, value_col="conf_attack_auc")
    if output_path:
        sig_df.to_csv(output_path, index=False)
    return sig_df


def run_significance_tests_lira(results_df, output_path=None):
    """Exploratory paired t-tests on LiRA AUC (defense vs none), if column exists."""
    if "lira_attack_auc" not in results_df.columns:
        empty = pd.DataFrame()
        if output_path:
            empty.to_csv(output_path, index=False)
        return empty
    sig_df = significance_vs_no_defense(results_df, value_col="lira_attack_auc")
    if output_path:
        sig_df.to_csv(output_path, index=False)
    return sig_df


def bootstrap_ci_over_seeds(
    df,
    group_cols=("dataset", "model", "defense"),
    metric_cols=None,
    n_resamples=1000,
    confidence=0.95,
    random_state=0,
):
    """
    Bootstrap 95% CIs by resampling seeds with replacement within each group.
    Separate from exploratory t-test p-values (see significance*.csv).
    """
    if metric_cols is None:
        metric_cols = [
            "test_accuracy",
            "conf_attack_auc",
            "lira_attack_auc",
            "ece_test",
        ]
    metric_cols = [m for m in metric_cols if m in df.columns]
    rng = np.random.RandomState(random_state)
    alpha = (1.0 - confidence) / 2.0
    lo_q, hi_q = alpha, 1.0 - alpha
    rows = []
    for tup, g in df.groupby(list(group_cols)):
        ds, model, defense = tup
        seeds = g["seed"].values
        if len(seeds) < 2:
            for m in metric_cols:
                mu = float(np.nanmean(g[m].values))
                rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "defense": defense,
                        "metric": m,
                        "mean": mu,
                        "std": float(np.nanstd(g[m].values, ddof=1))
                        if len(g) > 1
                        else 0.0,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n_seeds": int(len(g)),
                    }
                )
            continue
        for m in metric_cols:
            vals = g.set_index("seed")[m].sort_index().values
            boots = []
            for _ in range(n_resamples):
                idx = rng.randint(0, len(vals), size=len(vals))
                boots.append(np.nanmean(vals[idx]))
            boots = np.array(boots)
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "defense": defense,
                    "metric": m,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "ci_low": float(np.quantile(boots, lo_q)),
                    "ci_high": float(np.quantile(boots, hi_q)),
                    "n_seeds": int(len(vals)),
                }
            )
    return pd.DataFrame(rows)


def run_bootstrap_summary(results_df, output_path=None, bootstrap_cfg=None):
    if bootstrap_cfg is None:
        bootstrap_cfg = {"n_resamples": 1000, "confidence": 0.95}
    n_res = int(bootstrap_cfg.get("n_resamples", 1000))
    ci = float(bootstrap_cfg.get("confidence", 0.95))
    bdf = bootstrap_ci_over_seeds(
        results_df,
        n_resamples=n_res,
        confidence=ci,
    )
    if output_path:
        bdf.to_csv(output_path, index=False)
    return bdf
