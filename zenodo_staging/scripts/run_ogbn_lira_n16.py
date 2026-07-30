"""
Tier A shadow-scaling credibility: ogbn-arxiv GraphSAGE undefended (none),
3 seeds, n_shadows=16 (classic LiRA-scale attack budget).

Answers reviewer concern that n_shadows=2--4 Volume nulls are underpowered.
Bypasses experiment_config_ogbn.yaml's n_shadows=2 cap.
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
N_SHADOWS = 16
OUT_CSV = "results/ogbn_lira_n16_3seed.csv"
OUT_JSON = "results/ogbn_lira_n16_3seed_summary.json"
OUT_LOG = "results/ogbn_lira_n16.log"


def main():
    os.environ["PRIVACYGNN_CONFIG"] = "experiment_config_ogbn.yaml"
    cfg = dict(load_config())
    ensure_dirs(cfg)
    cfg["lira"] = {"n_shadows": N_SHADOWS}
    cfg["attacks"] = ["confidence", "lira"]
    cfg["large_graph_use_official_split"] = True
    # Force CPU to match n=2/n=4 ogbn credibility protocol (NeighborLoader path).
    device = torch.device("cpu")
    cfg["device"] = "cpu"

    rows = []
    if os.path.isfile(OUT_CSV):
        rows = pd.read_csv(OUT_CSV).to_dict("records")
        done = {(int(r["seed"]), r["defense"]) for r in rows if int(r.get("n_shadows_run", 0)) == N_SHADOWS}
        print(f"resume: {len(done)} done on {device}", flush=True)
    else:
        done = set()

    print(
        f"Tier A: ogbn-arxiv GraphSAGE none | n_shadows={N_SHADOWS} | seeds={SEEDS} | device={device}",
        flush=True,
    )

    for seed in SEEDS:
        name, params = "none", {}
        if (seed, name) in done:
            print(f"skip {seed} {name}", flush=True)
            continue
        print(f"RUN seed={seed} defense={name} n_shadows={N_SHADOWS}", flush=True)
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
        row["n_shadows_run"] = N_SHADOWS
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
            "tpr01",
            row.get("lira_tpr_at_0.01_fpr"),
            "wall",
            row["wall_seconds"],
            flush=True,
        )

    df = pd.DataFrame(rows)
    means = (
        df.groupby("defense")[
            [
                "test_accuracy",
                "conf_attack_auc",
                "lira_attack_auc",
                "lira_tpr_at_0.01_fpr",
                "wall_seconds",
            ]
        ]
        .mean()
        .round(4)
        .to_dict()
    )
    per_seed = (
        df[["seed", "test_accuracy", "conf_attack_auc", "lira_attack_auc", "wall_seconds"]]
        .round(4)
        .to_dict("records")
    )
    # Compare to existing n=2 / n=4 credibility numbers for the paper table.
    summary = {
        "tier": "A",
        "purpose": "Shadow-scaling credibility: classic LiRA n_shadows=16 on Volume null (ogbn-arxiv)",
        "n_shadows": N_SHADOWS,
        "seeds": SEEDS,
        "defenses": ["none"],
        "device": str(device),
        "split_protocol": "ogb_official",
        "means": means,
        "per_seed": per_seed,
        "compare_to_prior": {
            "n2_systems_grid_none_lira": 0.502,
            "n4_credibility_none_lira": 0.503,
            "n4_credibility_none_acc": 0.666,
        },
        "note": (
            "Official OGB split; undefended GraphSAGE only (Tier A). "
            "Addresses reviewer concern that n_shadows<=4 Volume nulls are underpowered vs classic LiRA 16+."
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {OUT_CSV} and {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
