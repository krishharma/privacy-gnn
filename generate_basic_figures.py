"""
Basic result visuals from summary.csv (mean over seeds).
Run: python generate_basic_figures.py
Outputs to figures/ as PNG only (300 dpi).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from config import load_config
    cfg = load_config()
    FIG_DIR = cfg["figures_dir"]
    SUMMARY = os.path.join(cfg["results_dir"], "summary.csv")
except Exception:
    _ROOT = os.path.dirname(os.path.abspath(__file__))
    FIG_DIR = os.path.join(_ROOT, "figures")
    SUMMARY = os.path.join(_ROOT, "results", "summary.csv")

os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.35,
})

SYN_PREFIX = "synthetic_"
DATASET_SHORT = {
    "synthetic_high_sparse": "H hi / sparse",
    "synthetic_high_medium": "H hi / med",
    "synthetic_high_dense": "H hi / dense",
    "synthetic_low_sparse": "Lo / sparse",
    "synthetic_low_medium": "Lo / med",
    "synthetic_low_dense": "Lo / dense",
}

MODELS = ["LogReg", "MLP", "GCN", "GraphSAGE"]
COLORS = {"LogReg": "#4477AA", "MLP": "#66CCEE", "GCN": "#CC6677", "GraphSAGE": "#882255"}


def save(fig, name):
    path_png = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path_png}")


def main():
    df = pd.read_csv(SUMMARY)
    syn = df[df["dataset"].str.startswith(SYN_PREFIX)].copy()
    if syn.empty:
        print("No synthetic rows in summary.csv; nothing to plot.")
        return

    none_only = syn[syn["defense"] == "none"].copy()
    none_only["short_ds"] = none_only["dataset"].map(DATASET_SHORT).fillna(none_only["dataset"])
    ds_order = [d for d in DATASET_SHORT if d in none_only["dataset"].values]

    # ----- Figure 1: grouped bars — attack AUC by model across synthetic settings (no defense) -----
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ds_order))
    w = 0.2
    for i, m in enumerate(MODELS):
        sub = none_only[none_only["model"] == m].set_index("dataset").reindex(ds_order)
        vals = sub["attack_auc_mean"].values
        err = sub["attack_auc_std"].fillna(0).values
        ax.bar(x + (i - 1.5) * w, vals, width=w, yerr=err, capsize=2,
               label=m, color=COLORS[m], edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_SHORT[d] for d in ds_order], rotation=25, ha="right")
    ax.set_ylabel("Confidence attack AUC")
    ax.set_xlabel("Synthetic graph setting")
    ax.set_title("Membership leakage (no defense): models compared")
    ax.legend(loc="upper left", ncol=2, framealpha=0.95)
    ax.set_ylim(0.42, 0.68)
    fig.tight_layout()
    save(fig, "basic_fig1_attack_auc_models_no_defense")

    # ----- Figure 2: GCN only — same x-axis, single model -----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sub = none_only[none_only["model"] == "GCN"].set_index("dataset").reindex(ds_order)
    ax.bar(range(len(ds_order)), sub["attack_auc_mean"].values, yerr=sub["attack_auc_std"].values,
           capsize=3, color="#CC6677", edgecolor="#333333", linewidth=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(ds_order)))
    ax.set_xticklabels([DATASET_SHORT[d] for d in ds_order], rotation=25, ha="right")
    ax.set_ylabel("Confidence attack AUC")
    ax.set_title("GCN only: leakage vs graph setting (no defense)")
    ax.set_ylim(0.42, 0.68)
    fig.tight_layout()
    save(fig, "basic_fig2_gcn_attack_auc_by_setting")

    # ----- Figure 3: GCN on low-homophily medium — defenses -----
    ds_target = "synthetic_low_medium"
    gcn_lm = syn[(syn["dataset"] == ds_target) & (syn["model"] == "GCN")].copy()
    if not gcn_lm.empty:
        order = ["none", "dropedge", "label_smoothing", "early_stopping",
                 "confidence_masking", "edge_sparsification"]
        gcn_lm = gcn_lm.set_index("defense")
        gcn_lm = gcn_lm.reindex([d for d in order if d in gcn_lm.index])
        labels = [d.replace("_", " ").title() if d != "none" else "None" for d in gcn_lm.index]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ypos = np.arange(len(gcn_lm))
        ax.barh(ypos, gcn_lm["attack_auc_mean"].values,
                xerr=gcn_lm["attack_auc_std"].fillna(0).values, capsize=2,
                color="#4C72B0", edgecolor="white")
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Confidence attack AUC")
        ax.set_title(f"GCN defenses: {DATASET_SHORT.get(ds_target, ds_target)}")
        ax.set_xlim(0.45, 0.62)
        fig.tight_layout()
        save(fig, "basic_fig3_gcn_defenses_low_med")

    # ----- Figure 4: simple heatmap — models × settings (no defense) -----
    pivot = none_only.pivot_table(
        index="model", columns="dataset", values="attack_auc_mean", aggfunc="mean"
    )
    pivot = pivot.reindex(index=MODELS, columns=ds_order)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0.48, vmax=0.62)
    ax.set_xticks(range(len(ds_order)))
    ax.set_xticklabels([DATASET_SHORT[d] for d in ds_order], rotation=30, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title("Attack AUC heatmap (no defense, darker = more leakage)")
    for i in range(len(MODELS)):
        for j in range(len(ds_order)):
            v = pivot.values[i, j]
            if np.isnan(v):
                t = "—"
            else:
                t = f"{v:.2f}"
            ax.text(j, i, t, ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Attack AUC", shrink=0.85)
    fig.tight_layout()
    save(fig, "basic_fig4_heatmap_models_x_setting")

    print(f"\nDone. Figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
