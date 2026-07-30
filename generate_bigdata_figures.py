"""
Generate IEEE BigData paper figures from results/all_results.csv.
Figures: discovery heatmap, main defense comparison, Pareto, ablation, LTE note.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8,
    }
)

try:
    from config import load_config

    _config = load_config()
    FIG_DIR = _config["figures_dir"]
    _results_path = os.path.join(_config["results_dir"], "all_results.csv")
except Exception:
    _ROOT = os.path.dirname(os.path.abspath(__file__))
    FIG_DIR = os.path.join(_ROOT, "figures")
    _results_path = os.path.join(_ROOT, "results", "all_results.csv")

os.makedirs(FIG_DIR, exist_ok=True)
df = pd.read_csv(_results_path)
df["dataset"] = df["dataset"].astype(str).str.strip()
df["model"] = df["model"].astype(str).str.strip()
df["defense"] = df["defense"].astype(str).str.strip()

PRIV = "lira_attack_auc" if "lira_attack_auc" in df.columns and df["lira_attack_auc"].notna().any() else "conf_attack_auc"


def savefig(fig, name):
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}")


def fig_discovery():
    """No-defense attack AUROC across models × synthetic regimes."""
    syn = df[(df["defense"] == "none") & (df["dataset"].str.startswith("synthetic_"))]
    if syn.empty:
        return
    g = syn.groupby(["dataset", "model"])[PRIV].mean().unstack("model")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(g, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, vmin=0.45, vmax=0.7)
    ax.set_title(f"Discovery: no-defense {PRIV.replace('_', ' ')}")
    ax.set_xlabel("Model")
    ax.set_ylabel("Synthetic regime")
    savefig(fig, "fig_discovery_heatmap")


def fig_main_defenses():
    """GCN/SAGE defense comparison on citation + adverse synthetic."""
    focus_ds = ["Cora", "Citeseer", "PubMed", "synthetic_low_sparse"]
    defs = ["none", "dropedge", "label_smoothing", "lbp", "gtd", "sami"]
    sub = df[(df["dataset"].isin(focus_ds)) & (df["model"].isin(["GCN", "GraphSAGE"])) & (df["defense"].isin(defs))]
    if sub.empty:
        return
    g = sub.groupby(["dataset", "model", "defense"]).agg(
        acc=("test_accuracy", "mean"),
        priv=(PRIV, "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in zip(axes, ["priv", "acc"], [f"Privacy ({PRIV})", "Utility (test acc)"]):
        pivot = g.pivot_table(index="defense", columns=["dataset", "model"], values=metric)
        pivot = pivot.reindex([d for d in defs if d in pivot.index])
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis" if metric == "acc" else "magma_r", ax=ax)
        ax.set_title(title)
    fig.suptitle("Main defense comparison (GCN / GraphSAGE)", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_main_defenses")


def fig_pareto():
    """Utility–privacy scatter for GNN defenses."""
    defs = ["none", "dropedge", "label_smoothing", "lbp", "gtd", "sami", "advreg", "sami_temp_only"]
    sub = df[(df["model"].isin(["GCN", "GraphSAGE"])) & (df["defense"].isin(defs))]
    if sub.empty:
        return
    g = sub.groupby(["dataset", "model", "defense"]).agg(
        acc=("test_accuracy", "mean"),
        priv=(PRIV, "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    for defense, part in g.groupby("defense"):
        ax.scatter(part["priv"], part["acc"], label=defense, s=40, alpha=0.8)
    ax.axvline(0.5, color="gray", ls="--", lw=1, label="chance")
    ax.set_xlabel(f"Attack AUROC ({PRIV}) — lower is better")
    ax.set_ylabel("Test accuracy — higher is better")
    ax.set_title("Utility–privacy Pareto (ideal: upper-left)")
    ax.legend(loc="best", fontsize=7, ncol=2)
    savefig(fig, "fig_pareto")


def fig_ablation():
    """Ablation on Cora + synthetic_low_sparse for GCN."""
    ablations = ["sami", "sami_no_lte", "sami_no_adv", "sami_no_gate", "sami_temp_only", "advreg", "none"]
    sub = df[
        (df["dataset"].isin(["Cora", "synthetic_low_sparse"]))
        & (df["model"] == "GCN")
        & (df["defense"].isin(ablations))
    ]
    if sub.empty:
        return
    g = sub.groupby(["dataset", "defense"])[PRIV].mean().unstack("dataset")
    g = g.reindex([a for a in ablations if a in g.index])
    fig, ax = plt.subplots(figsize=(7, 4))
    g.plot(kind="bar", ax=ax)
    ax.set_ylabel(PRIV.replace("_", " "))
    ax.set_title("SAMI ablations (GCN)")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.legend(title="dataset")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    savefig(fig, "fig_ablation")


def fig_citation_rq1():
    """Feature-only vs GNN leakage on citation graphs (no defense)."""
    sub = df[(df["defense"] == "none") & (df["dataset"].isin(["Cora", "Citeseer", "PubMed"]))]
    if sub.empty:
        return
    g = sub.groupby(["dataset", "model"])[PRIV].mean().unstack("model")
    fig, ax = plt.subplots(figsize=(7, 4))
    g.plot(kind="bar", ax=ax)
    ax.set_ylabel(PRIV.replace("_", " "))
    ax.set_title("RQ1: citation benchmarks (no defense)")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    plt.xticks(rotation=0)
    fig.tight_layout()
    savefig(fig, "fig_citation_rq1")


if __name__ == "__main__":
    print(f"Reading {_results_path} ({len(df)} rows); privacy metric={PRIV}")
    fig_discovery()
    fig_citation_rq1()
    fig_main_defenses()
    fig_pareto()
    fig_ablation()
    print("Done.")
