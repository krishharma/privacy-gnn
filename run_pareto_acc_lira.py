"""Multi-panel Acc vs LiRA Pareto matching paper tables."""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import pandas as pd

# Paper-facing means (match tables in ieee_privacy_gnn.tex)
PANELS = {
    "Cora": {
        "none": (0.877, 0.567),
        "gtd": (0.876, 0.538),
        "lbp": (0.701, 0.645),
        "maskarmor": (0.877, 0.580),
        "sami": (0.874, 0.510),
        "dp": (0.784, 0.509),  # Acc-tuned naive DP-SGD (vacuous ε; not GAP)
    },
    "Citeseer": {
        "none": (0.740, 0.596),
        "gtd": (0.739, 0.551),
        "lbp": (0.621, 0.530),
        "maskarmor": (0.740, 0.660),
        "sami": (0.731, 0.537),  # locked config (≈ selected 0.538)
    },
    "Chameleon": {
        "none": (0.449, 0.623),
        "gtd": (0.475, 0.581),
        "lbp": (0.389, 0.544),
        "maskarmor": (0.449, 0.774),
        "sami": (0.460, 0.551),
    },
    "Actor": {
        "none": (0.335, 0.605),
        "gtd": (0.339, 0.585),
        "lbp": (0.289, 0.543),
        "maskarmor": (0.335, 0.797),
        "sami": (0.317, 0.552),  # locked
        "sami_sel": (0.306, 0.537),  # val high-σ
    },
}

MARKERS = {
    "none": "o",
    "gtd": "s",
    "lbp": "^",
    "maskarmor": "D",
    "sami": "*",
    "sami_sel": "P",
    "dp": "X",
}
COLORS = {
    "none": "#333333",
    "gtd": "#4C78A8",
    "lbp": "#F58518",
    "maskarmor": "#E45756",
    "sami": "#54A24B",
    "sami_sel": "#B279A2",
    "dp": "#72B7B2",
}
LABELS = {
    "none": "none",
    "gtd": "GTD",
    "lbp": "LBP",
    "maskarmor": "MaskArmor",
    "sami": "SAMI",
    "sami_sel": "SAMI (val)",
    "dp": "DP-SGD†",
}


def main():
    rows = []
    for ds, pts in PANELS.items():
        for d, (acc, lira) in pts.items():
            rows.append({"dataset": ds, "defense": d, "acc": acc, "lira": lira})
    pd.DataFrame(rows).to_csv("results/pareto_acc_lira.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8))
    for ax, ds in zip(axes.ravel(), ["Cora", "Citeseer", "Chameleon", "Actor"]):
        pts = PANELS[ds]
        for d, (acc, lira) in pts.items():
            ax.scatter(
                acc,
                lira,
                marker=MARKERS[d],
                c=COLORS[d],
                s=120 if d.startswith("sami") else 55,
                label=LABELS[d],
                zorder=3,
                edgecolors="k",
                linewidths=0.35,
            )
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("Test Acc ↑", fontsize=8)
        ax.set_ylabel("LiRA AUROC ↓", fontsize=8)
        ax.axhline(0.5, color="#888", ls=":", lw=0.9)
        ax.grid(True, ls=":", alpha=0.35)
        # Highlight ideal corner (high Acc, low LiRA)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=MARKERS[d],
            color="w",
            markerfacecolor=COLORS[d],
            markeredgecolor="k",
            markersize=8,
            label=LABELS[d],
        )
        for d in ["none", "gtd", "lbp", "maskarmor", "sami", "sami_sel", "dp"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.suptitle(
        "Acc–LiRA joint view. †DP-SGD = Acc-tuned naive clip+noise (vacuous ε; not GAP).",
        fontsize=9,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    for path in [
        "figures/fig_pareto_acc_lira.png",
        "paper/fig_pareto_acc_lira.png",
        "paper/paper_visuals/fig_pareto_acc_lira.png",
    ]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote Pareto figure")


if __name__ == "__main__":
    main()
