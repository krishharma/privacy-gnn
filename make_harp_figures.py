"""
Build HARP paper figures from results/harp_baselines.csv / harp_means.csv.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
PV = os.path.join(ROOT, "paper", "paper_visuals")


def _save(fig, name: str):
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(PV, exist_ok=True)
    for d in (FIG, PV):
        fig.savefig(os.path.join(d, name + ".png"), dpi=220, bbox_inches="tight")
        fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_harp_pareto(means: pd.DataFrame):
    """Acc vs LiRA; marker size ~ noise mass. Cora GraphSAGE."""
    sub = means[(means.dataset == "Cora") & (means.model == "GraphSAGE")].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    colors = {
        "none": "#444444",
        "lbp": "#c1121f",
        "gtd": "#f4a261",
        "sami": "#2a9d8f",
        "harp": "#1d3557",
        "harp_k0": "#457b9d",
        "harp_k2": "#a8dadc",
        "harp_uniform": "#e9c46a",
        "harp_release_only": "#6d6875",
    }
    for _, r in sub.iterrows():
        mass = r.get("noise_mass", np.nan)
        if mass != mass or mass <= 0:
            s = 60
        else:
            s = 40 + 180 * float(mass) / float(sub["noise_mass"].max())
        ax.scatter(
            r["lira_attack_auc"],
            r["test_accuracy"],
            s=s,
            c=colors.get(r["defense"], "#333"),
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
            label=r["defense"],
        )
        ax.annotate(r["defense"], (r["lira_attack_auc"], r["test_accuracy"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axvline(0.5, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("LiRA AUROC (lower better)")
    ax.set_ylabel("Test accuracy (higher better)")
    ax.set_title("Cora GraphSAGE: Acc–LiRA (marker ∝ noise mass)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig_harp_pareto")


def fig_noise_mass(means: pd.DataFrame):
    """Noise mass and Acc for none/lbp/sami/harp across primary datasets."""
    defs = ["lbp", "sami", "harp"]
    dss = ["Cora", "Citeseer", "Chameleon", "Actor"]
    sub = means[(means.model == "GraphSAGE") & (means.defense.isin(defs)) & (means.dataset.isin(dss))]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    x = np.arange(len(dss))
    w = 0.25
    for i, d in enumerate(defs):
        vals = []
        for ds in dss:
            row = sub[(sub.dataset == ds) & (sub.defense == d)]
            vals.append(float(row.noise_mass.iloc[0]) if len(row) else np.nan)
        axes[0].bar(x + (i - 1) * w, vals, width=w, label=d)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dss, fontsize=8)
    axes[0].set_ylabel("Noise mass Σσ_v")
    axes[0].set_title("Release noise mass")
    axes[0].legend(fontsize=7, frameon=False)

    for i, d in enumerate(defs):
        vals = []
        for ds in dss:
            row = sub[(sub.dataset == ds) & (sub.defense == d)]
            vals.append(float(row.test_accuracy.iloc[0]) if len(row) else np.nan)
        axes[1].bar(x + (i - 1) * w, vals, width=w, label=d)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dss, fontsize=8)
    axes[1].set_ylabel("Test Acc")
    axes[1].set_title("Utility")
    axes[1].legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig_harp_efficiency")


def fig_harp_schematic():
    """Simple schematic: seeds → k-hop → selective noise."""
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, 1.0, 2.2, 1.4, "LTE risk\n$r_v$"),
        (3.0, 1.0, 2.2, 1.4, "Top-risk\nseeds"),
        (5.7, 1.0, 2.2, 1.4, "$k$-hop\nexpand"),
        (8.4, 1.0, 1.4, 1.4, "Noise\nonly there"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor="#edf2f4",
                                   edgecolor="#1d3557", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)
    for x0 in (2.5, 5.2, 7.9):
        ax.annotate("", xy=(x0 + 0.45, 1.7), xytext=(x0, 1.7),
                    arrowprops=dict(arrowstyle="->", color="#1d3557"))
    ax.set_title("HARP: Hop-Aware Risk-conditioned Privacy", fontsize=10, pad=8)
    fig.tight_layout()
    _save(fig, "fig_harp_schematic")


def main():
    fig_harp_schematic()
    means_path = os.path.join(RES, "harp_means.csv")
    if not os.path.isfile(means_path):
        print("harp_means.csv not ready; schematic only")
        return
    means = pd.read_csv(means_path)
    fig_harp_pareto(means)
    fig_noise_mass(means)


if __name__ == "__main__":
    main()
