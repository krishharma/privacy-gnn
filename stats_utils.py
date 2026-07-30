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


def run_significance_tests(results_df, output_path=None, adjustment="holm"):
    """
    Paired t-tests (defense vs none) on conf_attack_auc with Holm/BH adjustment.
    """
    sig_df = significance_vs_no_defense(results_df, value_col="conf_attack_auc")
    sig_df = apply_multiple_comparison(sig_df, method=adjustment)
    if output_path:
        sig_df.to_csv(output_path, index=False)
    return sig_df


def run_significance_tests_lira(results_df, output_path=None, adjustment="holm"):
    """Paired t-tests on LiRA AUC (defense vs none) with multiple-comparison control."""
    if "lira_attack_auc" not in results_df.columns:
        empty = pd.DataFrame()
        if output_path:
            empty.to_csv(output_path, index=False)
        return empty
    sig_df = significance_vs_no_defense(results_df, value_col="lira_attack_auc")
    sig_df = apply_multiple_comparison(sig_df, method=adjustment)
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
            "lira_tpr_at_0.01_fpr",
            "mlp_phi_attack_auc",
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


def bootstrap_delta_ci(
    df,
    value_col="lira_attack_auc",
    baseline="none",
    defense="sami",
    group_cols=("dataset", "model"),
    n_resamples=2000,
    confidence=0.95,
    random_state=0,
):
    """
    Bootstrap CIs on the paired effect Δ = mean(defense - baseline) itself,
    by resampling seed indices and recomputing the mean difference each time.
    """
    rng = np.random.RandomState(random_state)
    alpha = (1.0 - confidence) / 2.0
    lo_q, hi_q = alpha, 1.0 - alpha
    rows = []
    for tup, g in df.groupby(list(group_cols)):
        if len(tup) == 2:
            ds, model = tup
        else:
            ds, model = tup[0], tup[1]
        base = g[g["defense"] == baseline].set_index("seed")[value_col]
        other = g[g["defense"] == defense].set_index("seed")[value_col]
        common = base.index.intersection(other.index).sort_values()
        if len(common) < 2:
            continue
        a = base.reindex(common).values.astype(float)
        b = other.reindex(common).values.astype(float)
        deltas = b - a
        boots = []
        for _ in range(n_resamples):
            idx = rng.randint(0, len(deltas), size=len(deltas))
            boots.append(np.nanmean(deltas[idx]))
        boots = np.asarray(boots)
        rows.append(
            {
                "dataset": ds,
                "model": model,
                "defense": defense,
                "baseline": baseline,
                "metric": value_col,
                "delta_mean": float(np.mean(deltas)),
                "delta_std": float(np.std(deltas, ddof=1)),
                "ci_low": float(np.quantile(boots, lo_q)),
                "ci_high": float(np.quantile(boots, hi_q)),
                "n_seeds": int(len(deltas)),
            }
        )
    return pd.DataFrame(rows)


def power_analysis_paired(
    delta_mean,
    delta_std,
    alpha=0.05,
    power=0.8,
):
    """
    Approximate number of paired seeds needed to detect |delta_mean| at the
    given alpha/power under a normal paired-difference model (two-sided).
    Returns dict with n_required and observed_effect metadata.
    """
    from math import ceil

    # z_{1-α/2} and z_{power}
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    d = abs(float(delta_mean))
    s = float(delta_std)
    if d < 1e-12 or s < 1e-12:
        n_req = float("inf")
    else:
        n_req = ((z_alpha + z_power) * s / d) ** 2
    return {
        "delta_mean": float(delta_mean),
        "delta_std": float(delta_std),
        "alpha": float(alpha),
        "power": float(power),
        "n_seeds_required": float(n_req) if n_req != float("inf") else None,
        "n_seeds_required_ceil": int(ceil(n_req)) if n_req != float("inf") else None,
        "cohens_dz": float(d / s) if s > 1e-12 else None,
    }


def run_bootstrap_delta_summary(
    results_df,
    output_path=None,
    defenses=("sami", "lbp", "gtd", "label_smoothing", "dropedge"),
    metrics=("lira_attack_auc", "conf_attack_auc"),
    bootstrap_cfg=None,
):
    if bootstrap_cfg is None:
        bootstrap_cfg = {"n_resamples": 2000, "confidence": 0.95}
    frames = []
    for metric in metrics:
        if metric not in results_df.columns:
            continue
        for defense in defenses:
            frames.append(
                bootstrap_delta_ci(
                    results_df,
                    value_col=metric,
                    defense=defense,
                    n_resamples=int(bootstrap_cfg.get("n_resamples", 2000)),
                    confidence=float(bootstrap_cfg.get("confidence", 0.95)),
                )
            )
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if output_path:
        out.to_csv(output_path, index=False)
    return out


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


def holm_bonferroni(p_values):
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def benjamini_hochberg(p_values):
    """Benjamini–Hochberg FDR-adjusted p-values (same order as input)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * n / (rank + 1)
        prev = min(prev, val)
        adj[idx] = min(1.0, prev)
    return adj


def apply_multiple_comparison(sig_df, method="holm", p_col="p_value"):
    """Add p_adjusted column using Holm or BH across all rows in sig_df."""
    if sig_df is None or sig_df.empty or p_col not in sig_df.columns:
        return sig_df
    out = sig_df.copy()
    raw = out[p_col].values.astype(float)
    if method == "bh":
        out["p_adjusted"] = benjamini_hochberg(raw)
        out["adjustment"] = "bh"
    else:
        out["p_adjusted"] = holm_bonferroni(raw)
        out["adjustment"] = "holm"
    if "diff" in out.columns:
        out["effect_size_delta"] = out["diff"]
    return out


def confirmatory_pairwise(
    df,
    value_col="lira_attack_auc",
    utility_col="test_accuracy",
    defenses=("sami", "lbp", "gtd", "dropedge", "label_smoothing"),
    baseline="none",
):
    """Confirmatory paired comparisons of each defense vs baseline across seeds."""
    out = []
    for (ds, model), g in df.groupby(["dataset", "model"]):
        base = g[g["defense"] == baseline]
        if base.empty or value_col not in g.columns:
            continue
        base_priv = base.set_index("seed")[value_col]
        base_util = base.set_index("seed")[utility_col] if utility_col in g.columns else None
        for defense in defenses:
            if defense == baseline:
                continue
            other = g[g["defense"] == defense]
            if other.empty:
                continue
            o_priv = other.set_index("seed")[value_col]
            common = base_priv.index.intersection(o_priv.index)
            if len(common) < 2:
                continue
            a = base_priv.reindex(common).dropna()
            b = o_priv.reindex(common).dropna()
            common2 = a.index.intersection(b.index)
            if len(common2) < 2:
                continue
            av = a.reindex(common2).values
            bv = b.reindex(common2).values
            t, p = stats.ttest_rel(av, bv, alternative="two-sided")
            row = {
                "dataset": ds,
                "model": model,
                "defense": defense,
                "baseline": baseline,
                "metric": value_col,
                "mean_baseline": float(np.mean(av)),
                "mean_defense": float(np.mean(bv)),
                "diff": float(np.mean(bv - av)),
                "t_stat": float(t),
                "p_value": float(p),
                "n_seeds": int(len(common2)),
            }
            if base_util is not None and utility_col in other.columns:
                bu = base_util.reindex(common2).dropna()
                ou = other.set_index("seed")[utility_col].reindex(common2).dropna()
                c3 = bu.index.intersection(ou.index)
                if len(c3) >= 1:
                    row["util_diff"] = float(
                        np.mean(ou.reindex(c3).values - bu.reindex(c3).values)
                    )
                    row["util_baseline"] = float(np.mean(bu.reindex(c3).values))
                    row["util_defense"] = float(np.mean(ou.reindex(c3).values))
            out.append(row)
    return pd.DataFrame(out)


def run_confirmatory_tests(
    results_df, output_path=None, adjustment="holm", value_col="lira_attack_auc"
):
    """Confirmatory SAMI vs baselines with adjusted p-values and utility deltas."""
    if value_col not in results_df.columns:
        value_col = "conf_attack_auc"
    cdf = confirmatory_pairwise(results_df, value_col=value_col)
    cdf = apply_multiple_comparison(cdf, method=adjustment)
    if output_path:
        cdf.to_csv(output_path, index=False)
    return cdf
