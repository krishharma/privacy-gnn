"""
Systems figure: audit cost vs graph size (train time, LiRA shadow wall, QPS).
Uses frozen ogbn / citation timing; writes CSV + PNG.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_config


def main():
    cfg = load_config()
    res = cfg["results_dir"]
    figdir = cfg.get("figures_dir", "figures")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs("paper/paper_visuals", exist_ok=True)

    rows = []
    # Citation-ish from core / timing
    timing = os.path.join(res, "timing_overhead.csv")
    if os.path.isfile(timing):
        tdf = pd.read_csv(timing)
        for _, r in tdf.iterrows():
            rows.append(
                {
                    "dataset": r.get("dataset", "citation"),
                    "n_nodes": r.get("n_nodes", np.nan),
                    "defense": r.get("defense", "none"),
                    "train_seconds": r.get("train_seconds", r.get("wall_seconds", np.nan)),
                    "source": "timing_overhead",
                }
            )

    # Actor
    ab = os.path.join(res, "actor_baselines.csv")
    if os.path.isfile(ab):
        adf = pd.read_csv(ab)
        cols = [c for c in ["train_seconds", "wall", "wall_seconds"] if c in adf.columns]
        g = adf.groupby("defense")[cols].mean(numeric_only=True)
        for d, r in g.iterrows():
            rows.append(
                {
                    "dataset": "Actor",
                    "n_nodes": 7600,
                    "defense": d,
                    "train_seconds": float(r.get("train_seconds", np.nan)),
                    "source": "actor_baselines",
                }
            )

    # ogbn volume
    ov = os.path.join(res, "ogbn_volume_results.csv")
    if os.path.isfile(ov):
        odf = pd.read_csv(ov)
        g = odf.groupby("defense")[["train_seconds", "wall_seconds", "peak_rss_mb"]].mean(
            numeric_only=True
        )
        for d, r in g.iterrows():
            rows.append(
                {
                    "dataset": "ogbn-arxiv",
                    "n_nodes": 169343,
                    "defense": d,
                    "train_seconds": float(r.get("train_seconds", np.nan)),
                    "wall_seconds": float(r.get("wall_seconds", np.nan)),
                    "peak_rss_mb": float(r.get("peak_rss_mb", np.nan)),
                    "source": "ogbn_volume",
                }
            )

    # n4 extension if present
    n4 = os.path.join(res, "ogbn_lira_n4_3seed.csv")
    if os.path.isfile(n4):
        ndf = pd.read_csv(n4)
        g = ndf.groupby("defense")[["train_seconds", "wall_seconds"]].mean(numeric_only=True)
        for d, r in g.iterrows():
            rows.append(
                {
                    "dataset": "ogbn-arxiv_n4",
                    "n_nodes": 169343,
                    "defense": d,
                    "train_seconds": float(r.get("train_seconds", np.nan)),
                    "wall_seconds": float(r.get("wall_seconds", np.nan)),
                    "n_shadows": 4,
                    "source": "ogbn_lira_n4",
                }
            )

    qps = None
    sj = os.path.join(res, "ogbn_systems.json")
    if os.path.isfile(sj):
        qps = json.load(open(sj)).get("api_qps_approx")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(res, "systems_audit_cost.csv"), index=False)

    # Figure: train seconds vs n for none/sami
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for defense, marker in [("none", "o"), ("sami", "s")]:
        sub = out[out.defense == defense].dropna(subset=["n_nodes", "train_seconds"])
        if sub.empty:
            continue
        # one point per dataset (mean)
        g = sub.groupby("dataset")[["n_nodes", "train_seconds"]].mean()
        ax.scatter(g["n_nodes"], g["train_seconds"], marker=marker, s=60, label=defense)
        for ds, r in g.iterrows():
            ax.annotate(ds.replace("ogbn-arxiv", "ogbn"), (r["n_nodes"], r["train_seconds"]), fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nodes (log)")
    ax.set_ylabel("Train seconds (log)")
    title = "Audit train cost vs scale"
    if qps:
        title += f"  |  ogbn QPS≈{int(qps)}"
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    for path in [
        os.path.join(figdir, "fig_systems_audit_cost.png"),
        os.path.join("paper/paper_visuals", "fig_systems_audit_cost.png"),
        os.path.join("paper", "fig_systems_audit_cost.png"),
    ]:
        fig.savefig(path, dpi=200)
    plt.close(fig)
    print(out.to_string(index=False))
    print("wrote systems_audit_cost + figure; qps=", qps)


if __name__ == "__main__":
    main()
