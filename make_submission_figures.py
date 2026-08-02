#!/usr/bin/env python3
"""Final submission figures from locked CSVs (n_sh=16 headline + ExactFrac evidence)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUTS = [os.path.join(ROOT, "figures"), os.path.join(ROOT, "paper"), os.path.join(ROOT, "paper", "paper_visuals")]

C = {
    "none": "#2f2f2f",
    "lbp": "#b33a3a",
    "eq": "#c48a00",
    "harp": "#0b6e4f",
    "gap": "#1b4965",
    "mg": "#6b4c7a",
    "ink": "#1a1a1a",
}


def _save(fig, name: str):
    for d in OUTS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{name}.png"), dpi=260, bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(d, f"{name}.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def fig_headline():
    m = pd.read_csv(os.path.join(RES, "harp_headline_nsh16_means.csv")).set_index("tag")
    frac = pd.read_csv(os.path.join(RES, "harp_frac_sweep_5seed.csv"))
    og = pd.read_csv(os.path.join(RES, "harp_ogbn.csv"))
    og = og[og.defense.isin(["none", "lbp", "harp"])]

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.1), gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    order = ["none", "lbp_eq", "memguard", "harp_locked", "gap_s3"]
    labels = ["none", "eq-mass\nLBP", "MemGuard", "HARP", "GAP-agg"]
    ece = [float(m.loc[t, "ECE"]) for t in order]
    colors = [C["none"], C["eq"], C["mg"], C["harp"], C["gap"]]
    x = np.arange(len(order))
    ax.bar(x, ece, color=colors, width=0.72)
    # mark ExactFrac-feasible
    for i, t in enumerate(order):
        ef = float(m.loc[t, "ExactFrac"])
        if ef >= 0.59:
            ax.text(i, ece[i] + 0.012, "EF✓", ha="center", fontsize=7, color=C["harp"], fontweight="bold")
        else:
            ax.text(i, ece[i] + 0.012, "EF✗", ha="center", fontsize=7, color=C["lbp"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel("Test ECE ↓", fontsize=9)
    ax.set_title("A. ECE at $n_{sh}{=}16$ (EF = ExactFrac)", fontsize=9, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(ece) * 1.35)

    ax = axes[1]
    g = frac.groupby("target_frac")[["test_accuracy", "noise_mass"]].mean()
    fr = g.index.values.astype(float)
    ax.plot(fr, g["test_accuracy"], "o-", color=C["harp"], lw=1.9, ms=5.5, label="Acc")
    ax.axvline(0.40, color=C["harp"], ls=":", lw=0.9, alpha=0.7)
    ax.set_xlabel("Protected Frac", fontsize=8)
    ax.set_ylabel("Test Acc ↑", color=C["harp"], fontsize=9)
    ax.tick_params(axis="y", labelcolor=C["harp"])
    ax2 = ax.twinx()
    ax2.plot(fr, g["noise_mass"], "s--", color=C["lbp"], lw=1.4, ms=4.5)
    ax2.set_ylabel("Mass ↓", color=C["lbp"], fontsize=9)
    ax2.tick_params(axis="y", labelcolor=C["lbp"])
    ax.set_title("B. Acc–Mass dial (locked Frac${=}0.40$)", fontsize=9, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)

    ax = axes[2]
    means = og.groupby("defense")["test_accuracy"].mean()
    stds = og.groupby("defense")["test_accuracy"].std()
    order2 = ["none", "lbp", "harp"]
    labels2 = ["none", "strong LBP", "HARP"]
    vals = [float(means[d]) for d in order2]
    errs = [float(stds[d]) if d in stds and not np.isnan(stds[d]) else 0.0 for d in order2]
    cols = [C["none"], C["lbp"], C["harp"]]
    ax.bar(np.arange(3), vals, yerr=errs, color=cols, width=0.72, capsize=2.5)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels2, fontsize=8)
    ax.set_ylabel("ogbn-arxiv Acc ↑", fontsize=9)
    ax.set_title("C. Volume Acc vs. strong LBP", fontsize=9, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 0.85)

    fig.suptitle("HARP systems headline (ExactFrac-constrained serving)", fontsize=10.5, fontweight="bold", y=1.02)
    _save(fig, "fig_harp_headline")


def fig_pareto_annotate():
    """Refresh ExactFrac Pareto if source exists; else skip gracefully."""
    src = os.path.join(RES, "harp_exactfrac_pareto.csv")
    if not os.path.isfile(src):
        # keep existing figure; just stamp a small companion Acc-LiRA at n16
        m = pd.read_csv(os.path.join(RES, "harp_headline_nsh16_means.csv"))
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        styles = {
            "none": (C["none"], "o", "none (EF=1)"),
            "harp_locked": (C["harp"], "D", "HARP (EF=0.60)"),
            "gap_s3": (C["gap"], "s", "GAP-agg (EF=1)"),
            "lbp_eq": (C["eq"], "^", "eq-mass LBP (EF=0)"),
            "memguard": (C["mg"], "v", "MemGuard (EF=0)"),
            "lbp_strong": (C["lbp"], "x", "strong LBP (EF=0)"),
        }
        for tag, (col, mk, lab) in styles.items():
            r = m[m.tag == tag].iloc[0]
            ax.scatter(r["LiRA"], r["Acc"], c=col, marker=mk, s=70, label=lab, zorder=3)
            ax.errorbar(r["LiRA"], r["Acc"], xerr=r["LiRA_std"], yerr=r["Acc_std"],
                        fmt="none", ecolor=col, alpha=0.45, lw=0.9)
        ax.axvline(0.5, color="#999", ls="--", lw=0.7)
        ax.set_xlabel("LiRA AUROC ↓ ($n_{sh}{=}16$)", fontsize=9)
        ax.set_ylabel("Test Acc ↑", fontsize=9)
        ax.set_title("Cora Acc–LiRA at headline budget", fontsize=10, loc="left", fontweight="bold")
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0.52, 0.86)
        ax.set_ylim(0.68, 0.90)
        # shade ExactFrac-feasible
        ax.text(0.535, 0.885, "EF-feasible: none, GAP, HARP", fontsize=7, color=C["harp"])
        _save(fig, "fig_harp_nsh16_scatter")
        return
    print("pareto csv present; leaving fig_harp_exactfrac_pareto as-is")


def fig_cache_story():
    rep = pd.read_csv(os.path.join(RES, "harp_replay_flicker.csv"))
    g = rep.groupby("policy")[["measured_exactfrac_requery", "top1_flicker", "threshold_flicker"]].mean()
    # normalize names
    order = [p for p in ["none", "gap", "harp", "lbp"] if p in g.index]
    labels = {"none": "none", "gap": "GAP-agg", "harp": "HARP", "lbp": "LBP"}
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = np.arange(len(order))
    w = 0.25
    ax.bar(x - w, [g.loc[p, "measured_exactfrac_requery"] for p in order], w, color=C["harp"], label="Bit-equal re-query")
    ax.bar(x, [g.loc[p, "top1_flicker"] for p in order], w, color=C["eq"], label="Top-1 flip")
    ax.bar(x + w, [g.loc[p, "threshold_flicker"] for p in order], w, color=C["lbp"], label="Thresh. flip")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[p] for p in order], fontsize=8)
    ax.set_ylabel("Rate", fontsize=9)
    ax.set_title("Measured ExactFrac / flicker (not definitional)", fontsize=9.5, loc="left", fontweight="bold")
    ax.legend(fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.15)
    _save(fig, "fig_harp_cache_veracity")


if __name__ == "__main__":
    fig_headline()
    fig_pareto_annotate()
    fig_cache_story()
    print("FIGURES DONE")
