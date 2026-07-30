"""
ogbn-arxiv LiRA credibility: none + SAMI, 3 seeds, n_shadows=4.
Bypasses run_ogbn_volume.py's min(n_shadows, yaml=2) cap.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import torch

from config import ensure_dirs, load_config
from experiment import run_one

SEEDS = [42, 123, 456]
SAMI = {
    "lam": 0.5,
    "use_lte": True,
    "use_gate": False,
    "arch_aware": True,
    "warmup_epochs": 3,
    "entropy_coef": 0.05,
    "noise_scale": 0.1,
    "budget_B": 0.0,
}
OUT_CSV = "results/ogbn_lira_n4_3seed.csv"
OUT_JSON = "results/ogbn_lira_n4_3seed_summary.json"


def main():
    os.environ["PRIVACYGNN_CONFIG"] = "experiment_config_ogbn.yaml"
    cfg = dict(load_config())
    ensure_dirs(cfg)
    cfg["lira"] = {"n_shadows": 4}
    cfg["attacks"] = ["confidence", "lira"]
    cfg["large_graph_use_official_split"] = True
    device = torch.device(cfg.get("device", "cpu"))

    rows = []
    if os.path.isfile(OUT_CSV):
        rows = pd.read_csv(OUT_CSV).to_dict("records")
        done = {(int(r["seed"]), r["defense"]) for r in rows}
        print(f"resume: {len(done)} done", flush=True)
    else:
        done = set()

    for seed in SEEDS:
        for name, params in [("none", {}), ("sami", SAMI)]:
            if (seed, name) in done:
                print(f"skip {seed} {name}", flush=True)
                continue
            print(f"RUN seed={seed} defense={name} n_shadows=4", flush=True)
            t0 = time.time()
            row = run_one(
                "ogbn-arxiv",
                "GraphSAGE",
                name,
                params,
                seed,
                config=cfg,
                device=device,
            )
            row["wall_seconds"] = round(time.time() - t0, 2)
            row["n_shadows_run"] = 4
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            print(
                seed,
                name,
                "acc",
                row["test_accuracy"],
                "conf",
                row["conf_attack_auc"],
                "lira",
                row["lira_attack_auc"],
                "wall",
                row["wall_seconds"],
                flush=True,
            )

    df = pd.DataFrame(rows)
    summary = {
        "n_shadows": 4,
        "seeds": SEEDS,
        "means": df.groupby("defense")[
            ["test_accuracy", "conf_attack_auc", "lira_attack_auc", "wall_seconds"]
        ]
        .mean()
        .round(4)
        .to_dict(),
        "note": "Official OGB split; LiRA n=4 × 3 seeds for none+SAMI. Negative control credibility check.",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
