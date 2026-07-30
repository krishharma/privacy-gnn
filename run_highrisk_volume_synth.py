"""
High-risk Volume×leakage cell via experiment.run_one.

Cell: GCN, n=3000, h≈0.15, sparse, SNR=0.05, hidden=128, epochs=200.
Probe showed conf≈0.80 with Acc≈0.28 (Actor-like imperfect Acc + high MIA).
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
    cfg = load_config()
    ensure_dirs(cfg)
    cfg = dict(cfg)
    cfg["lira"] = {"n_shadows": 4}
    cfg["attacks"] = ["confidence", "lira"]
    cfg["training"] = dict(
        cfg.get("training", {}),
        epochs=200,
        hidden=128,
        lr=0.01,
        weight_decay=5e-4,
    )

    dataset = "synthetic_low_sparse_snr0.05_n3000_h0.15"
    seeds = [42, 123, 456]
    sami = {
        "lam": 0.5,
        "use_lte": True,
        "use_gate": False,
        "arch_aware": True,
        "warmup_epochs": 5,
        "noise_scale": 0.35,  # locked SAMI config (Table samicfg)
        "entropy_coef": 0.05,
    }
    rows = []
    device = torch.device(cfg.get("device", "cpu"))
    for seed in seeds:
        for name, params in [("none", {}), ("sami", sami)]:
            t0 = time.time()
            row = run_one(
                dataset, "GCN", name, params, int(seed), config=cfg, device=device
            )
            row["volume_cell"] = "highrisk_n3k_gcn"
            row["wall_seconds"] = round(time.time() - t0, 2)
            rows.append(row)
            print(
                f"{dataset} seed={seed} {name}: acc={row['test_accuracy']} "
                f"conf={row['conf_attack_auc']} lira={row['lira_attack_auc']} "
                f"gap={row.get('gen_gap')} wall={row['wall_seconds']}s",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(
                os.path.join(cfg["results_dir"], "volume_highrisk_synth.csv"), index=False
            )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("defense")[
            ["test_accuracy", "conf_attack_auc", "lira_attack_auc", "gen_gap"]
        ]
        .mean()
        .round(4)
        .to_dict()
    )
    out = {
        "purpose": "Volume×leakage: n=3k GCN low-h/sparse/low-SNR, imperfect Acc",
        "dataset": dataset,
        "n_seeds": len(seeds),
        "n_shadows": 4,
        "summary_means": summary,
    }
    with open(
        os.path.join(cfg["results_dir"], "volume_highrisk_synth_summary.json"), "w"
    ) as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
