"""
Generate a full poster-ready visual suite in ../Final Visuals.

Run from repo root:
  python privacy-gnn/generate_poster_visuals.py
"""
from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

_ROOT = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_ROOT, "..", "Final Visuals")
_SUMMARY = os.path.join(_ROOT, "results", "summary.csv")
_SIG = os.path.join(_ROOT, "results", "significance.csv")

C = {
    "gcn": "#0F4C5C",
    "sage": "#5C4A72",
    "logreg": "#6B7C85",
    "mlp": "#9AA5AA",
    "accent": "#C45C3E",
    "muted": "#94A3B8",
    "line": "#334155",
    "bg": "#FAFAF8",
    "grid": "#E2E8F0",
    "ok": "#2D7D46",
    "bad": "#AF3E2E",
}

MODELS = ["LogReg", "MLP", "GCN", "GraphSAGE"]
MODEL_COLOR = {"LogReg": C["logreg"], "MLP": C["mlp"], "GCN": C["gcn"], "GraphSAGE": C["sage"]}

DS_ORDER = [
    "synthetic_high_dense",
    "synthetic_high_medium",
    "synthetic_high_sparse",
    "synthetic_low_dense",
    "synthetic_low_medium",
    "synthetic_low_sparse",
]
DS_SHORT = {
    "synthetic_high_dense": "Hi · dense",
    "synthetic_high_medium": "Hi · med",
    "synthetic_high_sparse": "Hi · sparse",
    "synthetic_low_dense": "Lo · dense",
    "synthetic_low_medium": "Lo · med",
    "synthetic_low_sparse": "Lo · sparse",
}


def _parse_regime(ds: str) -> Tuple[str, str]:
    homo = "Low homophily" if "low" in ds else "High homophily"
    density = "Dense" if "dense" in ds else ("Medium" if "medium" in ds else "Sparse")
    return homo, density


def _rc() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11.5,
            "axes.titlesize": 15,
            "axes.titleweight": "600",
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "font.family": "sans-serif",
            "axes.facecolor": C["bg"],
            "figure.facecolor": "white",
            "axes.labelcolor": C["line"],
            "axes.titlecolor": "#0F172A",
            "text.color": C["line"],
        }
    )


def _style_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C["line"])
    ax.spines["bottom"].set_color(C["line"])
    ax.spines["left"].set_linewidth(1.05)
    ax.spines["bottom"].set_linewidth(1.05)
    ax.tick_params(colors=C["line"], width=1.0, length=4, pad=6)
    ax.set_axisbelow(True)
    if grid_axis in ("x", "y", "both"):
        ax.grid(axis=grid_axis, color=C["grid"], linewidth=0.9, alpha=0.95)


def _finish_layout(fig, *, left=0.08, right=0.98, bottom=0.14, top=0.9) -> None:
    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def _save(fig, name: str) -> None:
    os.makedirs(_OUT, exist_ok=True)
    base = os.path.join(_OUT, name)
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {base}.png")


def _poster_00_study_frame() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.7))
    ax.axis("off")

    boxes = [
        (0.03, 0.2, 0.22, 0.6, "Question", "How do GNN architecture and graph structure\nimpact membership leakage?"),
        (0.29, 0.2, 0.22, 0.6, "Data", "6 synthetic graph regimes:\n2 homophily levels × 3 densities"),
        (0.55, 0.2, 0.22, 0.6, "Models + Defenses", "4 models and 6 defenses\nmeasured over repeated runs"),
        (0.81, 0.2, 0.16, 0.6, "Outcomes", "Utility:\nTest Acc\n\nPrivacy:\nAttack AUC"),
    ]
    for x, y, w, h, title, body in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.2,
            edgecolor=C["line"],
            facecolor="#F8FAFC",
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.14, title, transform=ax.transAxes, fontsize=13.5, weight="600", color="#0F172A")
        ax.text(x + 0.02, y + h - 0.21, body, transform=ax.transAxes, fontsize=10.8, va="top")

    for x0, x1 in [(0.25, 0.29), (0.51, 0.55), (0.77, 0.81)]:
        ax.annotate("", xy=(x1, 0.5), xytext=(x0, 0.5), xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color=C["line"], lw=1.8))

    ax.set_title("Poster Overview: End-to-End Research Pipeline", pad=10)
    _finish_layout(fig, left=0.02, right=0.99, bottom=0.08, top=0.88)
    _save(fig, "poster_00_study_pipeline")


def _poster_01_attack_heatmap_none(syn: pd.DataFrame) -> None:
    none = syn[syn["defense"] == "none"]
    pivot = none.pivot_table(index="model", columns="dataset", values="attack_auc_mean", aggfunc="mean")
    pivot = pivot.reindex(index=MODELS, columns=DS_ORDER)
    fig, ax = plt.subplots(figsize=(11.4, 4.3))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0.48, vmax=0.62)
    ax.set_xticks(np.arange(len(DS_ORDER)))
    ax.set_xticklabels([DS_SHORT[d] for d in DS_ORDER], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10.6, color="#0F172A", fontweight="600")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Attack AUC")
    ax.set_title("No-Defense Leakage Map Across Models and Graph Regimes")
    ax.set_xlabel("Graph regime")
    _finish_layout(fig, left=0.08, right=0.98, bottom=0.26, top=0.88)
    _save(fig, "poster_01_leakage_heatmap_no_defense")


def _poster_02_privacy_utility_by_model(syn: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for model in MODELS:
        part = syn[syn["model"] == model]
        ax.scatter(
            part["test_acc_mean"],
            part["attack_auc_mean"],
            s=85,
            alpha=0.88,
            color=MODEL_COLOR[model],
            label=model,
            edgecolors="white",
            linewidths=0.9,
        )
    ax.axhline(0.5, color=C["muted"], linestyle="--", linewidth=1.3)
    ax.set_xlabel("Test accuracy (utility)")
    ax.set_ylabel("Attack AUC (privacy leakage)")
    ax.set_title("Privacy-Utility Landscape by Model (All Regimes and Defenses)")
    ax.legend(ncols=2, loc="upper right", frameon=False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    _style_axes(ax, grid_axis="both")
    _finish_layout(fig, left=0.11, right=0.98, bottom=0.14, top=0.9)
    _save(fig, "poster_02_privacy_utility_by_model")


def _poster_03_defense_delta_heatmap(syn: pd.DataFrame) -> None:
    baseline = syn[syn["defense"] == "none"][["dataset", "model", "attack_auc_mean"]].rename(
        columns={"attack_auc_mean": "auc_none"}
    )
    merged = syn.merge(baseline, on=["dataset", "model"], how="left")
    merged["delta_auc"] = merged["attack_auc_mean"] - merged["auc_none"]
    defenses = [d for d in sorted(syn["defense"].unique()) if d != "none"]
    grid = (
        merged[merged["defense"].isin(defenses)]
        .groupby(["model", "defense"], as_index=False)["delta_auc"]
        .mean()
        .pivot(index="model", columns="defense", values="delta_auc")
        .reindex(index=MODELS, columns=defenses)
    )

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    vmax = np.nanmax(np.abs(grid.values))
    im = ax.imshow(grid.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(defenses)))
    ax.set_xticklabels([d.replace("_", " ") for d in defenses], rotation=24, ha="right")
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid.values[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=9.8, fontweight="600")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("Delta Attack AUC vs no-defense")
    ax.set_title("Defense Effect on Leakage (Negative is Better Privacy)")
    _finish_layout(fig, left=0.09, right=0.98, bottom=0.28, top=0.88)
    _save(fig, "poster_03_defense_delta_heatmap")


def _poster_04_defense_ranked_bars(syn: pd.DataFrame) -> None:
    baseline = syn[syn["defense"] == "none"][["dataset", "model", "attack_auc_mean"]].rename(
        columns={"attack_auc_mean": "auc_none"}
    )
    merged = syn.merge(baseline, on=["dataset", "model"], how="left")
    merged["gain"] = merged["auc_none"] - merged["attack_auc_mean"]
    ranked = (
        merged[merged["defense"] != "none"]
        .groupby(["defense"], as_index=False)["gain"]
        .mean()
        .sort_values("gain", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    y = np.arange(len(ranked))
    colors = [C["ok"] if v > 0 else C["bad"] for v in ranked["gain"]]
    ax.barh(y, ranked["gain"], color=colors, height=0.62, edgecolor="white")
    ax.axvline(0.0, color=C["line"], linewidth=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([d.replace("_", " ").title() for d in ranked["defense"]])
    ax.invert_yaxis()
    ax.set_xlabel("Mean reduction in attack AUC vs no-defense")
    ax.set_title("Overall Defense Ranking Across Models and Regimes")
    _style_axes(ax, grid_axis="x")
    ax.grid(axis="y", visible=False)
    _finish_layout(fig, left=0.29, right=0.98, bottom=0.14, top=0.9)
    _save(fig, "poster_04_defense_ranking")


def _poster_05_attack_family_comparison(syn: pd.DataFrame) -> None:
    none = syn[syn["defense"] == "none"].copy()
    comp = none.groupby("model", as_index=False)[["attack_auc_mean", "thresh_auc_mean", "shadow_auc_mean"]].mean()
    comp = comp.set_index("model").reindex(MODELS)
    fig, ax = plt.subplots(figsize=(10.1, 5.2))
    x = np.arange(len(MODELS))
    w = 0.24
    ax.bar(x - w, comp["attack_auc_mean"], width=w, color=C["gcn"], label="Confidence attack", edgecolor="white")
    ax.bar(x, comp["thresh_auc_mean"], width=w, color=C["accent"], label="Threshold attack", edgecolor="white")
    ax.bar(x + w, comp["shadow_auc_mean"], width=w, color=C["sage"], label="Shadow attack", edgecolor="white")
    ax.axhline(0.5, color=C["muted"], linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Attack AUC (higher = more leakage)")
    ax.set_title("Attack Family Comparison (No Defense, Averaged Across Regimes)")
    ax.legend(loc="upper left", ncols=1, frameon=False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    _style_axes(ax, grid_axis="y")
    _finish_layout(fig, left=0.09, right=0.98, bottom=0.14, top=0.89)
    _save(fig, "poster_05_attack_family_comparison")


def _poster_06_homophily_density_effects(syn: pd.DataFrame) -> None:
    tmp = syn.copy()
    parsed = tmp["dataset"].apply(_parse_regime)
    tmp["homophily"] = parsed.apply(lambda t: t[0])
    tmp["density"] = parsed.apply(lambda t: t[1])
    agg = (
        tmp.groupby(["homophily", "density"], as_index=False)[["attack_auc_mean", "test_acc_mean"]]
        .mean()
        .sort_values(["homophily", "density"])
    )
    density_order = ["Dense", "Medium", "Sparse"]
    x = np.arange(len(density_order))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "attack_auc_mean", "Attack AUC"),
        (axes[1], "test_acc_mean", "Test accuracy"),
    ]:
        for label, color, marker in [
            ("High homophily", C["gcn"], "o"),
            ("Low homophily", C["accent"], "s"),
        ]:
            part = agg[agg["homophily"] == label].set_index("density").reindex(density_order)
            ax.plot(
                x,
                part[metric].values,
                marker=marker,
                color=color,
                linewidth=2.2,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.9,
                label=label,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(density_order)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        _style_axes(ax, grid_axis="y")
    axes[0].set_title("Main effects on privacy")
    axes[1].set_title("Main effects on utility")
    axes[1].legend(loc="lower left", frameon=False)
    fig.suptitle("Homophily and Density Shape Both Privacy and Utility", y=0.97, fontsize=16, fontweight="600")
    _finish_layout(fig, left=0.07, right=0.98, bottom=0.16, top=0.84)
    _save(fig, "poster_06_homophily_density_effects")


def _poster_07_calibration_vs_leakage(syn: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for model in MODELS:
        part = syn[syn["model"] == model]
        ax.scatter(
            part["ece_test_mean"],
            part["attack_auc_mean"],
            s=90,
            color=MODEL_COLOR[model],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.9,
            label=model,
        )
    coeff = np.polyfit(syn["ece_test_mean"].values, syn["attack_auc_mean"].values, deg=1)
    xs = np.linspace(float(syn["ece_test_mean"].min()), float(syn["ece_test_mean"].max()), 120)
    ax.plot(xs, coeff[0] * xs + coeff[1], color=C["line"], linestyle="--", linewidth=1.6, label="Linear trend")
    ax.axhline(0.5, color=C["muted"], linestyle=":", linewidth=1.2)
    ax.set_xlabel("ECE on test set (calibration error)")
    ax.set_ylabel("Attack AUC")
    ax.set_title("Calibration vs Privacy Leakage")
    ax.legend(loc="upper left", ncols=2, frameon=False)
    _style_axes(ax, grid_axis="both")
    _finish_layout(fig, left=0.11, right=0.98, bottom=0.14, top=0.9)
    _save(fig, "poster_07_calibration_vs_leakage")


def _poster_08_significance_map() -> None:
    if not os.path.exists(_SIG):
        return
    sig = pd.read_csv(_SIG)
    sig["neglog10_p"] = -np.log10(np.clip(sig["p_value"].values, 1e-12, 1.0))
    defenses = [d for d in sorted(sig["defense"].unique()) if d != "none"]
    agg = (
        sig.groupby(["model", "defense"], as_index=False)["neglog10_p"]
        .mean()
        .pivot(index="model", columns="defense", values="neglog10_p")
        .reindex(index=["GCN", "GraphSAGE"], columns=defenses)
    )
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    vmax = max(2.0, float(np.nanmax(agg.values)))
    im = ax.imshow(agg.values, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(defenses)))
    ax.set_xticklabels([d.replace("_", " ") for d in defenses], rotation=24, ha="right")
    ax.set_yticks(np.arange(agg.shape[0]))
    ax.set_yticklabels(list(agg.index))
    for i in range(agg.shape[0]):
        for j in range(agg.shape[1]):
            val = agg.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, fontweight="600")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("Mean -log10(p)")
    ax.set_title("Significance Strength of Defense Effects (Lower p is Stronger)")
    _finish_layout(fig, left=0.08, right=0.98, bottom=0.3, top=0.86)
    _save(fig, "poster_08_significance_heatmap")


def _poster_09_dataset_small_multiples(syn: pd.DataFrame) -> None:
    gnn = syn[syn["model"].isin(["GCN", "GraphSAGE"])].copy()
    defenses = ["none", "dropedge", "label_smoothing", "early_stopping", "confidence_masking", "edge_sparsification"]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2), sharey=True)
    axes = axes.flatten()
    x = np.arange(len(defenses))
    for i, ds in enumerate(DS_ORDER):
        ax = axes[i]
        for model, color in [("GCN", C["gcn"]), ("GraphSAGE", C["sage"])]:
            part = gnn[(gnn["dataset"] == ds) & (gnn["model"] == model)].set_index("defense").reindex(defenses)
            ax.plot(
                x,
                part["attack_auc_mean"].values,
                marker="o",
                linewidth=2.0,
                color=color,
                markersize=5.8,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=model,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("_", " ") for d in defenses], rotation=28, ha="right")
        ax.set_title(DS_SHORT[ds], fontsize=11.5, pad=6)
        ax.axhline(0.5, color=C["muted"], linestyle="--", linewidth=1.0)
        _style_axes(ax, grid_axis="y")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Defense Behavior Across Regimes (GCN vs GraphSAGE)", y=0.99, fontsize=16, fontweight="600")
    _finish_layout(fig, left=0.06, right=0.99, bottom=0.19, top=0.9)
    _save(fig, "poster_09_regime_small_multiples")


def _poster_10_key_metrics_table(syn: pd.DataFrame) -> None:
    none = syn[syn["defense"] == "none"]
    best_priv = (
        syn.groupby(["model", "defense"], as_index=False)["attack_auc_mean"]
        .mean()
        .sort_values(["model", "attack_auc_mean"])
        .groupby("model", as_index=False)
        .first()
        .rename(columns={"defense": "best_privacy_defense", "attack_auc_mean": "best_defense_auc"})
    )
    baseline = none.groupby("model", as_index=False)[["attack_auc_mean", "test_acc_mean"]].mean()
    table = baseline.merge(best_priv, on="model", how="left").set_index("model").reindex(MODELS)
    table["auc_reduction"] = table["attack_auc_mean"] - table["best_defense_auc"]

    fig, ax = plt.subplots(figsize=(10.8, 3.8))
    ax.axis("off")
    rows = []
    for model in MODELS:
        r = table.loc[model]
        rows.append(
            [
                model,
                f"{r['test_acc_mean']:.3f}",
                f"{r['attack_auc_mean']:.3f}",
                str(r["best_privacy_defense"]).replace("_", " ").title(),
                f"{r['best_defense_auc']:.3f}",
                f"{r['auc_reduction']:.3f}",
            ]
        )
    col_labels = [
        "Model",
        "No-defense\nUtility",
        "No-defense\nAttack AUC",
        "Best Privacy\nDefense",
        "Best-defense\nAttack AUC",
        "AUC\nReduction",
    ]
    table_artist = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(10)
    table_artist.scale(1.0, 1.85)
    for (r, c), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if r == 0:
            cell.set_facecolor("#E2E8F0")
            cell.set_text_props(weight="600", color="#0F172A")
        else:
            cell.set_facecolor("#F8FAFC" if r % 2 else "white")
    ax.set_title("Poster Summary Table: Baseline vs Best Privacy Defense", pad=12)
    _finish_layout(fig, left=0.03, right=0.97, bottom=0.08, top=0.82)
    _save(fig, "poster_10_summary_table")


def main() -> None:
    _rc()
    syn = pd.read_csv(_SUMMARY)
    syn = syn[syn["dataset"].str.startswith("synthetic_")].copy()

    _poster_00_study_frame()
    _poster_01_attack_heatmap_none(syn)
    _poster_02_privacy_utility_by_model(syn)
    _poster_03_defense_delta_heatmap(syn)
    _poster_04_defense_ranked_bars(syn)
    _poster_05_attack_family_comparison(syn)
    _poster_06_homophily_density_effects(syn)
    _poster_07_calibration_vs_leakage(syn)
    _poster_08_significance_map()
    _poster_09_dataset_small_multiples(syn)
    _poster_10_key_metrics_table(syn)

    print(f"\nDone. Poster visual suite saved to: {_OUT}")


if __name__ == "__main__":
    main()
