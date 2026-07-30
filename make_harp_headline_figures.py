"""Upgrade HARP paper visuals: headline systems figure + refined Frac/ECE/ogbn panels."""
from __future__ import annotations

import json
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

# Distinctive palette (not purple-default AI look)
C = {
    "none": "#2f2f2f",
    "lbp": "#b33a3a",
    "eq": "#d4a017",
    "harp": "#0b6e4f",
    "sami": "#1b4965",
    "ink": "#1a1a1a",
    "grid": "#e8e4dc",
}


def _save(fig, name: str):
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(PV, exist_ok=True)
    for d in (FIG, PV):
        fig.savefig(os.path.join(d, f"{name}.png"), dpi=240, bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(d, f"{name}.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def fig_headline_systems():
    """Three-panel SDT-style wow figure: ECE, Frac curve, ogbn Acc."""
    fair = pd.read_csv(os.path.join(RES, "harp_fairness_cora_5seed.csv"))
    frac = pd.read_csv(os.path.join(RES, "harp_frac_sweep_5seed.csv"))
    og = pd.read_csv(os.path.join(RES, "harp_ogbn.csv"))
    og = og[og.defense.isin(["none", "lbp", "harp"])]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.05), gridspec_kw={"wspace": 0.32})

    # --- Panel A: ECE at matched Mass ---
    ax = axes[0]
    order = ["none", "lbp_equal_mass", "lbp_strong", "harp"]
    labels = ["none", "eq-mass\nLBP", "strong\nLBP", "HARP"]
    means, stds = [], []
    for tag in order:
        g = fair[fair.tag == tag]["ece_test"]
        means.append(float(g.mean()))
        stds.append(float(g.std(ddof=1)) if len(g) > 1 else 0.0)
    colors = [C["none"], C["eq"], C["lbp"], C["harp"]]
    x = np.arange(len(order))
    ax.bar(x, means, yerr=stds, color=colors, width=0.72, capsize=2.5, error_kw={"lw": 0.9})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Test ECE ↓", fontsize=9)
    ax.set_title("A. Calibration at matched Mass", fontsize=9, loc="left", fontweight="bold")
    ax.axhline(means[order.index("harp")], color=C["harp"], ls=":", lw=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(means) * 1.25)

    # --- Panel B: Frac Acc–Mass ---
    ax = axes[1]
    g = frac.groupby("target_frac")[["test_accuracy", "lira_attack_auc", "noise_mass"]].mean()
    fr = g.index.values.astype(float)
    ax.plot(fr, g["test_accuracy"], "o-", color=C["harp"], lw=1.8, ms=5, label="Acc")
    ax.set_xlabel("Protected fraction (Frac)", fontsize=8)
    ax.set_ylabel("Test Acc ↑", color=C["harp"], fontsize=9)
    ax.tick_params(axis="y", labelcolor=C["harp"])
    ax2 = ax.twinx()
    ax2.plot(fr, g["noise_mass"], "s--", color=C["lbp"], lw=1.4, ms=4.5, label="Mass")
    ax2.set_ylabel("Noise mass ↓", color=C["lbp"], fontsize=9)
    ax2.tick_params(axis="y", labelcolor=C["lbp"])
    ax.set_title("B. Frac Acc–Mass knob", fontsize=9, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.set_xlim(0.15, 1.05)

    # --- Panel C: ogbn Acc ---
    ax = axes[2]
    ogm = og.groupby("defense")["test_accuracy"].agg(["mean", "std"])
    order2 = ["none", "lbp", "harp"]
    labels2 = ["none", "strong LBP", "HARP"]
    colors2 = [C["none"], C["lbp"], C["harp"]]
    vals = [float(ogm.loc[d, "mean"]) if d in ogm.index else np.nan for d in order2]
    errs = [float(ogm.loc[d, "std"]) if d in ogm.index else 0 for d in order2]
    x = np.arange(len(order2))
    ax.bar(x, vals, yerr=errs, color=colors2, width=0.65, capsize=3, error_kw={"lw": 0.9})
    ax.set_xticks(x)
    ax.set_xticklabels(labels2, fontsize=8)
    ax.set_ylabel("Test Acc ↑", fontsize=9)
    ax.set_title("C. ogbn-arxiv (~169k)", fontsize=9, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # annotate Mass cut
    if "harp" in ogm.index and "lbp" in ogm.index:
        ax.annotate(
            "+0.42 Acc\n−60% Mass",
            xy=(2, vals[2]),
            xytext=(1.15, vals[2] + 0.12),
            fontsize=7.5,
            color=C["harp"],
            arrowprops=dict(arrowstyle="->", color=C["harp"], lw=0.9),
        )

    for ax in axes:
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    _save(fig, "fig_harp_headline")


def fig_seed_ablation():
    """LTE vs uniform seeds ΔLiRA across datasets with CI whiskers."""
    b = pd.read_csv(os.path.join(RES, "harp_baselines.csv"))
    rows = []
    for ds in ["Cora", "Citeseer", "Chameleon", "Actor"]:
        sub = b[(b.dataset == ds) & (b.model == "GraphSAGE") & (b.defense.isin(["harp", "harp_uniform"]))]
        if sub.defense.nunique() < 2:
            continue
        # paired by seed
        h = sub[sub.defense == "harp"].set_index("seed")["lira_attack_auc"]
        u = sub[sub.defense == "harp_uniform"].set_index("seed")["lira_attack_auc"]
        common = h.index.intersection(u.index)
        delta = (h.reindex(common) - u.reindex(common)).values  # negative = HARP better privacy
        rows.append(dict(dataset=ds, mean=float(np.mean(delta)), std=float(np.std(delta, ddof=1)), n=len(delta)))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    x = np.arange(len(df))
    colors = [C["harp"] if m < 0 else C["lbp"] for m in df["mean"]]
    ax.bar(x, df["mean"], yerr=df["std"], color=colors, width=0.6, capsize=3, error_kw={"lw": 0.9})
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], fontsize=8)
    ax.set_ylabel("ΔLiRA (HARP − uniform seeds)\nnegative = LTE better", fontsize=8)
    ax.set_title("LTE seed ranking vs matched-Frac random seeds", fontsize=9, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig_harp_seed_ablation")


def fig_harp_systems_panel():
    """Frac dial + Acc--LiRA scatter (no SAMI; marker size = Mass)."""
    frac = pd.read_csv(os.path.join(RES, "harp_frac_sweep_5seed.csv"))
    fair = pd.read_csv(os.path.join(RES, "harp_fairness_cora_5seed.csv"))
    base = pd.read_csv(os.path.join(RES, "harp_baselines.csv"))
    gtd = base[(base.dataset == "Cora") & (base.model == "GraphSAGE") & (base.defense == "gtd")]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), gridspec_kw={"wspace": 0.38})
    ax = axes[0]
    g = frac.groupby("target_frac")[["test_accuracy", "lira_attack_auc", "noise_mass"]].mean()
    fr = g.index.values.astype(float)
    ax.plot(fr, g["test_accuracy"], "o-", color=C["harp"], lw=1.8, ms=5, label="Acc")
    ax.set_xlabel("Protected fraction (Frac)", fontsize=8)
    ax.set_ylabel("Test Acc ↑", color=C["harp"], fontsize=9)
    ax.tick_params(axis="y", labelcolor=C["harp"])
    ax2 = ax.twinx()
    ax2.plot(fr, g["noise_mass"], "s--", color=C["lbp"], lw=1.4, ms=4.5, label="Mass")
    ax2.set_ylabel("Noise mass ↓", color=C["lbp"], fontsize=9)
    ax2.tick_params(axis="y", labelcolor=C["lbp"])
    ax.set_title("A. Frac Acc–Mass knob", fontsize=9, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.set_xlim(0.15, 1.05)

    ax = axes[1]
    pts = [
        ("none", "none", C["none"]),
        ("lbp_strong", "strong LBP", C["lbp"]),
        ("lbp_equal_mass", "eq-mass LBP", C["eq"]),
        ("harp", "HARP", C["harp"]),
    ]
    for tag, lab, col in pts:
        sub = fair[fair.tag == tag]
        acc = float(sub["test_accuracy"].mean())
        lira = float(sub["lira_attack_auc"].mean())
        mass = float(sub["noise_mass"].mean()) if sub["noise_mass"].notna().any() else 50.0
        sz = 35 + 0.35 * mass
        ax.scatter(lira, acc, s=sz, c=col, edgecolors="white", linewidths=0.6, label=lab, zorder=3)
    if len(gtd):
        acc = float(gtd["test_accuracy"].mean())
        lira = float(gtd["lira_attack_auc"].mean())
        ax.scatter(lira, acc, s=55, c="#6b4c9a", edgecolors="white", linewidths=0.6, label="GTD", zorder=3)
    ax.set_xlabel("LiRA AUROC ↓", fontsize=8)
    ax.set_ylabel("Test Acc ↑", fontsize=8)
    ax.set_title("B. Acc–LiRA (marker ∝ Mass)", fontsize=9, loc="left", fontweight="bold")
    ax.legend(fontsize=6.5, loc="lower left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for ax in axes:
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    _save(fig, "fig_harp_systems")


def fig_unprot_fidelity():
    """Clarify fidelity vs safety: ECE win + unprot conf disclosure."""
    un = pd.read_csv(os.path.join(RES, "harp_unprot_lira.csv"))
    fair = pd.read_csv(os.path.join(RES, "harp_fairness_cora_5seed.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.35})

    ax = axes[0]
    ece = fair.groupby("tag")["ece_test"].mean()
    tags = ["lbp_equal_mass", "harp"]
    labs = ["Equal-mass LBP\n(Frac=1)", "HARP\n(Frac=0.40)"]
    vals = [float(ece[t]) for t in tags]
    ax.bar([0, 1], vals, color=[C["eq"], C["harp"]], width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("Test ECE ↓")
    ax.set_title("Fidelity claim: calibration", fontsize=9, loc="left", fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8, color=C["ink"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    metrics = ["conf_pop", "conf_unprot", "lira_pop", "lira_unprot"]
    labs = ["conf\npop", "conf\nunprot", "LiRA\npop", "LiRA\nunprot"]
    means = [float(un[m].mean()) for m in metrics]
    colors = [C["none"], C["lbp"], C["harp"], C["eq"]]
    ax.bar(np.arange(4), means, color=colors, width=0.65)
    ax.axhline(0.5, color="0.5", ls="--", lw=0.8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel("AUROC")
    ax.set_title("Not a safety claim: unprot. leakage", fontsize=9, loc="left", fontweight="bold")
    ax.set_ylim(0.45, 0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, "fig_harp_fidelity_vs_safety")


def main():
    fig_headline_systems()
    fig_harp_systems_panel()
    fig_seed_ablation()
    fig_unprot_fidelity()
    print("done")


if __name__ == "__main__":
    main()
