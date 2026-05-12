"""
Generate powerful, minimal visuals for project results into ../Final Visuals/.
Reads results/summary.csv (aggregated over seeds).

Run from repo:  python privacy-gnn/generate_final_visuals.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, MultipleLocator

_ROOT = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_ROOT, "..", "Final Visuals")
_SUMMARY = os.path.join(_ROOT, "results", "summary.csv")

# Cohesive palette: teal / clay / slate (readable on screen and print)
C = {
    "gc": "#0F4C5C",
    "sage": "#5C4A72",
    "lr": "#6B7C85",
    "mlp": "#9AA5AA",
    "accent": "#C45C3E",
    "muted": "#94A3B8",
    "line": "#334155",
    "bg": "#FAFAF8",
}

DS_ORDER = [
    "synthetic_high_dense",
    "synthetic_high_medium",
    "synthetic_high_sparse",
    "synthetic_low_dense",
    "synthetic_low_medium",
    "synthetic_low_sparse",
]
DS_LABEL = {
    "synthetic_high_dense": "High homophily · dense",
    "synthetic_high_medium": "High homophily · medium",
    "synthetic_high_sparse": "High homophily · sparse",
    "synthetic_low_dense": "Low homophily · dense",
    "synthetic_low_medium": "Low homophily · medium",
    "synthetic_low_sparse": "Low homophily · sparse",
}
DS_SHORT = {
    "synthetic_high_dense": "Hi · dense",
    "synthetic_high_medium": "Hi · med",
    "synthetic_high_sparse": "Hi · sparse",
    "synthetic_low_dense": "Lo · dense",
    "synthetic_low_medium": "Lo · med",
    "synthetic_low_sparse": "Lo · sparse",
}

MODELS = ["LogReg", "MLP", "GCN", "GraphSAGE"]
MODEL_COLOR = {"LogReg": C["lr"], "MLP": C["mlp"], "GCN": C["gc"], "GraphSAGE": C["sage"]}


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C["line"])
    ax.spines["bottom"].set_color(C["line"])
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(colors=C["line"], length=4, width=1.0, pad=6)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.9, alpha=0.9)


def _finish_layout(fig, *, left=0.1, right=0.98, bottom=0.16, top=0.9):
    fig.tight_layout(pad=1.15)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def _rc():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.titleweight": "600",
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "font.family": "sans-serif",
            "axes.facecolor": C["bg"],
            "figure.facecolor": "white",
            "axes.grid": False,
            "axes.labelcolor": C["line"],
            "axes.titlecolor": "#0F172A",
            "text.color": C["line"],
            "legend.frameon": False,
            "legend.borderaxespad": 0.0,
            "legend.handlelength": 1.5,
        }
    )


def save(fig, name: str) -> None:
    os.makedirs(_OUT, exist_ok=True)
    base = os.path.join(_OUT, name)
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {base}.png")


def main() -> None:
    _rc()
    df = pd.read_csv(_SUMMARY)
    syn = df[df["dataset"].str.startswith("synthetic_")].copy()

    # ----- 1: Headline — worst case for GCN (low homophily, sparse), no defense -----
    target = "synthetic_low_sparse"
    sub = syn[(syn["dataset"] == target) & (syn["defense"] == "none")].set_index("model").reindex(MODELS)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    x = np.arange(len(MODELS))
    colors = [MODEL_COLOR[m] for m in MODELS]
    bars = ax.bar(
        x,
        sub["attack_auc_mean"].values,
        yerr=sub["attack_auc_std"].fillna(0).values,
        color=colors,
        width=0.62,
        capsize=4,
        edgecolor="white",
        linewidth=1.2,
        error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": C["line"]},
    )
    ax.axhline(0.5, color=C["muted"], linestyle="--", linewidth=1.5, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Membership inference AUC")
    ax.set_title("Hardest setting: low homophily, sparse graph", pad=20)
    ax.text(
        0.5,
        0.985,
        "Higher AUC → easier to tell train vs. test members (weaker privacy)",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color=C["line"],
        style="italic",
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ymax = float(sub["attack_auc_mean"].max() + sub["attack_auc_std"].fillna(0).max() + 0.04)
    ax.set_ylim(0.45, min(0.72, ymax))
    ax.margins(x=0.06)
    _style_axes(ax)
    _finish_layout(fig, left=0.1, right=0.98, bottom=0.16, top=0.86)
    save(fig, "final_01_headline_worst_case_models")

    # ----- 2: GCN vs GraphSAGE across all six settings (no defense) -----
    none_only = syn[syn["defense"] == "none"].copy()
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    xs = np.arange(len(DS_ORDER))
    for model, sty in [("GCN", "-o"), ("GraphSAGE", "-s")]:
        row = none_only[none_only["model"] == model].set_index("dataset").reindex(DS_ORDER)
        ax.errorbar(
            xs,
            row["attack_auc_mean"].values,
            yerr=row["attack_auc_std"].fillna(0).values,
            label=model,
            color=MODEL_COLOR[model],
            markersize=9,
            linewidth=2.4,
            capsize=4,
            markeredgecolor="white",
            markeredgewidth=1,
        )
    ax.axhline(0.5, color=C["muted"], linestyle="--", linewidth=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([DS_SHORT[d] for d in DS_ORDER], rotation=22, ha="right")
    ax.set_ylabel("Attack AUC")
    ax.set_title("Two GNNs, same defenses off: leakage vs. graph regime", pad=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99))
    ax.set_ylim(0.44, 0.66)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.margins(x=0.03)
    _style_axes(ax)
    _finish_layout(fig, left=0.08, right=0.99, bottom=0.22, top=0.9)
    save(fig, "final_02_gcn_vs_graphsage_trajectory")

    # ----- 3: Defenses on GCN (low homophily, medium density) — where edge sparsification helps -----
    ds_lm = "synthetic_low_medium"
    gcn_lm = syn[(syn["dataset"] == ds_lm) & (syn["model"] == "GCN")].copy()
    order = [
        "none",
        "dropedge",
        "label_smoothing",
        "early_stopping",
        "confidence_masking",
        "edge_sparsification",
    ]
    gcn_lm = gcn_lm.set_index("defense")
    gcn_lm = gcn_lm.reindex([d for d in order if d in gcn_lm.index])
    labels = [("No defense" if d == "none" else d.replace("_", " ").title()) for d in gcn_lm.index]
    vals = gcn_lm["attack_auc_mean"].values
    errs = gcn_lm["attack_auc_std"].fillna(0).values
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    y = np.arange(len(gcn_lm))
    bar_cols = [C["accent"] if ix == "edge_sparsification" else C["gc"] for ix in gcn_lm.index]
    ax.barh(y, vals, xerr=errs, color=bar_cols, height=0.65, capsize=3, edgecolor="white", linewidth=1)
    ax.axvline(0.5, color=C["muted"], linestyle="--", linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Attack AUC (lower is better for privacy)")
    ax.set_title("GCN on low homophily · medium density: defense comparison", pad=10)
    ax.set_xlim(0.48, 0.58)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.9, alpha=0.9)
    ax.grid(axis="y", visible=False)
    _style_axes(ax)
    _finish_layout(fig, left=0.29, right=0.98, bottom=0.16, top=0.9)
    save(fig, "final_03_defenses_gcn_low_medium")

    # ----- 4: Privacy–utility (GCN, all settings × defenses) -----
    gcn_all = syn[syn["model"] == "GCN"].copy()
    gcn_all["homo"] = gcn_all["dataset"].apply(lambda d: "Low homophily" if "low" in str(d) else "High homophily")
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for label, color, marker in [
        ("High homophily", C["gc"], "o"),
        ("Low homophily", C["accent"], "s"),
    ]:
        part = gcn_all[gcn_all["homo"] == label]
        ax.scatter(
            part["test_acc_mean"],
            part["attack_auc_mean"],
            s=130,
            alpha=0.88,
            c=color,
            marker=marker,
            label=label,
            edgecolors="white",
            linewidths=1.1,
        )
    ax.axhline(0.5, color=C["muted"], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Test accuracy (utility)")
    ax.set_ylabel("Attack AUC (privacy leakage)")
    ax.set_title("GCN: every point is one (graph regime × defense)", pad=10)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.margins(x=0.06, y=0.08)
    _style_axes(ax)
    _finish_layout(fig, left=0.11, right=0.98, bottom=0.14, top=0.9)
    save(fig, "final_04_gcn_privacy_utility_cloud")

    # ----- 5: One-glance matrix — attack AUC, no defense, models × settings -----
    none_only = none_only.copy()
    pivot = none_only.pivot_table(index="model", columns="dataset", values="attack_auc_mean", aggfunc="mean")
    pivot = pivot.reindex(index=MODELS, columns=[d for d in DS_ORDER if d in pivot.columns])
    fig, ax = plt.subplots(figsize=(11.2, 4.3))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0.48, vmax=0.62)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([DS_SHORT[c] for c in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title("Attack AUC at a glance (no defense) — darker = more leakage", pad=12)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                t = "—"
            else:
                t = f"{v:.2f}"
            ax.text(j, i, t, ha="center", va="center", fontsize=12, fontweight="600", color="#0F172A")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.025)
    cbar.set_label("AUC")
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel("Synthetic graph regime")
    _finish_layout(fig, left=0.08, right=0.98, bottom=0.26, top=0.87)
    save(fig, "final_05_matrix_glance_no_defense")

    print(f"\nDone. Output directory: {_OUT}")


if __name__ == "__main__":
    main()
