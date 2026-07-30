"""
ogbn-arxiv *resplit* Volume cell: forces a leaky membership regime at scale.
Official OGB split is the near-chance negative control; 40/20/40 random resplit
is the conditional high-risk Volume audit (none vs SAMI, 3 seeds).
"""
from __future__ import annotations

import json
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
    cfg["large_graph_use_official_split"] = False  # 40/20/40 resplit → leakier MIA
    cfg["lira"] = {"n_shadows": 2}
    cfg["attacks"] = ["confidence", "lira"]
    cfg["training"] = dict(cfg.get("training", {}), epochs=20)

    seeds = [42, 123, 456]
    defenses = [
        ("none", {}),
        (
            "sami",
            {
                "lam": 0.5,
                "use_lte": True,
                "use_gate": False,
                "arch_aware": True,
                "warmup_epochs": 3,
                "entropy_coef": 0.05,
                "noise_scale": 0.1,
                "budget_B": 0.0,
            },
        ),
    ]
    rows = []
    device = torch.device(cfg.get("device", "cpu"))
    for seed in seeds:
        for name, params in defenses:
            t0 = time.time()
            row = run_one(
                "ogbn-arxiv",
                "GraphSAGE",
                name,
                params,
                int(seed),
                config=cfg,
                device=device,
            )
            row["split_mode"] = "random_40_20_40"
            row["wall_seconds"] = round(time.time() - t0, 2)
            rows.append(row)
            print(
                f"resplit seed={seed} {name}: acc={row['test_accuracy']} "
                f"conf={row['conf_attack_auc']} lira={row['lira_attack_auc']} "
                f"wall={row['wall_seconds']}s"
            )
            pd.DataFrame(rows).to_csv(
                os.path.join(cfg["results_dir"], "ogbn_resplit_results.csv"), index=False
            )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("defense")[["test_accuracy", "conf_attack_auc", "lira_attack_auc"]]
        .mean()
        .round(4)
        .to_dict()
    )
    out = {
        "purpose": "Leaky Volume cell: random resplit of ogbn-arxiv (contrast with official-split negative control).",
        "n_seeds": len(seeds),
        "n_shadows": 2,
        "summary_means": summary,
    }
    with open(os.path.join(cfg["results_dir"], "ogbn_resplit_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
