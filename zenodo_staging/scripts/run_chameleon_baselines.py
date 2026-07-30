"""
Second heterophilic real graph: Chameleon GraphSAGE baselines
(none / GTD / LBP / MaskArmor / locked SAMI), 5 seeds, n_shadows=4.
Downloads WikipediaNetwork if needed.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import torch

from config import ensure_dirs, load_config
from experiment import run_one

SEEDS = [42, 123, 456, 789, 1024]
LOCKED_SAMI = {
    "lam": 0.5,
    "use_lte": True,
    "use_gate": True,
    "arch_aware": True,
    "noise_scale": 0.35,
    "budget_B": 0.0,
    "warmup_epochs": 5,
    "entropy_coef": 0.05,
}
OUT = "results/chameleon_baselines.csv"
OUT_JSON = "results/chameleon_baselines_summary.json"


def main():
    cfg = dict(load_config())
    ensure_dirs(cfg)
    cfg["attacks"] = ["confidence", "lira"]
    cfg["lira"] = {"n_shadows": 4}
    device = torch.device(cfg.get("device", "cpu"))

    defenses = [
        ("none", {}),
        ("gtd", {"gamma": 1.0, "stage1_frac": 0.5, "pseudo_conf": 0.8}),
        ("lbp", {"scale": 0.3}),
        ("maskarmor", {"top_k": 1}),
        ("sami", LOCKED_SAMI),
    ]

    rows = []
    if os.path.isfile(OUT):
        rows = pd.read_csv(OUT).to_dict("records")
        done = {(int(r["seed"]), r["defense"]) for r in rows}
    else:
        done = set()

    for seed in SEEDS:
        for name, params in defenses:
            if (seed, name) in done:
                continue
            print(f"Chameleon seed={seed} {name}", flush=True)
            t0 = time.time()
            try:
                row = run_one(
                    "Chameleon", "GraphSAGE", name, params, seed, config=cfg, device=device
                )
            except Exception as e:
                print(f"FAIL {seed} {name}: {e}", flush=True)
                with open("results/chameleon_baselines_error.json", "w") as f:
                    json.dump({"error": str(e), "seed": seed, "defense": name}, f, indent=2)
                raise
            row["wall_seconds"] = round(time.time() - t0, 2)
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(
                seed,
                name,
                row["test_accuracy"],
                row["lira_attack_auc"],
                row["conf_attack_auc"],
                flush=True,
            )

    df = pd.DataFrame(rows)
    means = (
        df.groupby("defense")[
            ["test_accuracy", "conf_attack_auc", "lira_attack_auc", "lira_tpr_at_0.01_fpr", "train_seconds"]
        ]
        .mean()
        .round(4)
    )
    summary = {
        "dataset": "Chameleon",
        "model": "GraphSAGE",
        "n_shadows": 4,
        "seeds": SEEDS,
        "sami_params": LOCKED_SAMI,
        "means": means.to_dict(),
        "homophily": float(df["homophily"].mean()) if "homophily" in df.columns else None,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(means)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
