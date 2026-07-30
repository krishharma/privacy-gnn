"""
Large + heterophilic regime fill: arxiv-year (LINKX).

Same ~169k-node citation graph as ogbn-arxiv, but year-quantile labels
(edge homophily ~0.22). Undefended GraphSAGE, 3 seeds, n_shadows=4 —
matched attack budget to the Volume ladder / tab:scale.
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
N_SHADOWS = 4
OUT_CSV = "results/arxiv_year_lira_n4_3seed.csv"
OUT_JSON = "results/arxiv_year_lira_n4_3seed_summary.json"


def main():
    os.environ["PRIVACYGNN_CONFIG"] = "experiment_config_ogbn.yaml"
    cfg = dict(load_config())
    ensure_dirs(cfg)
    cfg["lira"] = {"n_shadows": N_SHADOWS}
    cfg["attacks"] = ["confidence", "lira"]
    # Canonical LINKX-style 50/25/25 (seed 0) for the target; shadows resplit.
    cfg["large_graph_use_official_split"] = True
    device = torch.device("cpu")
    cfg["device"] = "cpu"

    rows = []
    if os.path.isfile(OUT_CSV):
        rows = pd.read_csv(OUT_CSV).to_dict("records")
        done = {
            (int(r["seed"]), r["defense"])
            for r in rows
            if int(r.get("n_shadows_run", 0)) == N_SHADOWS
        }
        print(f"resume: {len(done)} done on {device}", flush=True)
    else:
        done = set()

    print(
        f"arxiv-year GraphSAGE none | n_shadows={N_SHADOWS} | seeds={SEEDS} | device={device}",
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
            "arxiv-year",
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
            "h",
            row.get("homophily"),
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
                "homophily",
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
        df[
            [
                "seed",
                "homophily",
                "test_accuracy",
                "conf_attack_auc",
                "lira_attack_auc",
                "wall_seconds",
            ]
        ]
        .round(4)
        .to_dict("records")
    )
    summary = {
        "purpose": "Fill large+heterophilic regime cell (matched Volume peer to ogbn-arxiv)",
        "dataset": "arxiv-year",
        "n_shadows": N_SHADOWS,
        "seeds": SEEDS,
        "defenses": ["none"],
        "device": str(device),
        "split_protocol": "linkx_50_25_25_seed0",
        "means": means,
        "per_seed": per_seed,
        "compare_to_ogbn_arxiv_none_n4": {
            "homophily": 0.654,
            "acc": 0.666,
            "lira": 0.503,
        },
        "note": (
            "Same citation graph as ogbn-arxiv; year-quantile labels (5 bins, LINKX). "
            "Canonical target split = fixed 50/25/25 seed 0; shadows use random resplits."
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {OUT_CSV} and {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
