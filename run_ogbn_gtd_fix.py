"""
Re-run ogbn official-split GTD (and optional LBP) after Volume-safe GTD fix.
Appends/updates rows in ogbn_volume_results.csv for defense==gtd.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import torch

from config import ensure_dirs, load_config
from experiment import run_one


def main():
    os.environ["PRIVACYGNN_CONFIG"] = "experiment_config_ogbn.yaml"
    cfg = load_config()
    ensure_dirs(cfg)
    cfg = dict(cfg)
    cfg["large_graph_use_official_split"] = True
    cfg["lira"] = {"n_shadows": 2}
    cfg["attacks"] = ["confidence", "lira"]

    path = os.path.join(cfg["results_dir"], "ogbn_volume_results.csv")
    old = pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()
    # Keep non-GTD rows; replace GTD
    if len(old):
        keep = old[old["defense"] != "gtd"].copy()
    else:
        keep = pd.DataFrame()

    seeds = [42, 123, 456]
    # Volume-safe GTD: supervised CE dominant (stage1_frac=1.0) — no unstable pseudo stage
    params = {"gamma": 0.5, "stage1_frac": 1.0, "pseudo_conf": 0.9}
    rows = []
    device = torch.device(cfg.get("device", "cpu"))
    for seed in seeds:
        t0 = time.time()
        row = run_one(
            "ogbn-arxiv",
            "GraphSAGE",
            "gtd",
            params,
            int(seed),
            config=cfg,
            device=device,
        )
        row["wall_seconds"] = round(time.time() - t0, 2)
        row["gtd_volume_mode"] = "stage1_only_ce"
        rows.append(row)
        print(
            f"gtd-fixed seed={seed}: acc={row['test_accuracy']} conf={row['conf_attack_auc']} "
            f"lira={row['lira_attack_auc']} wall={row['wall_seconds']}s"
        )

    new_gtd = pd.DataFrame(rows)
    out = pd.concat([keep, new_gtd], ignore_index=True)
    out.to_csv(path, index=False)
    print(out.groupby("defense")[["test_accuracy", "conf_attack_auc", "lira_attack_auc"]].mean())


if __name__ == "__main__":
    main()
