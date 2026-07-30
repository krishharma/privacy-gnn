"""
Actor GraphSAGE baselines: none, SAMI (locked config), GTD, LBP, MaskArmor.
5 seeds, n_shadows=4. Writes results/actor_baselines.csv + summary JSON.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import torch

from config import ensure_dirs, load_config
from experiment import run_one

# Locked SAMI config used across datasets (Cora flagship / Actor / stress).
# noise_scale=0.35 matches sami_gtd_framing.json; selected on Cora val Acc only
# for the flagship cell — Actor uses the same locked weights (no Actor retune).
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


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    cfg = dict(cfg)
    cfg["lira"] = {"n_shadows": 4}
    cfg["attacks"] = ["confidence", "lira", "threshold"]

    seeds = [42, 123, 456, 789, 1024]
    defenses = [
        ("none", {}),
        ("sami", LOCKED_SAMI),
        ("gtd", {"gamma": 1.0, "stage1_frac": 0.5, "pseudo_conf": 0.8}),
        ("lbp", {"scale": 0.3}),
        ("maskarmor", {"top_k": 1}),
    ]
    rows = []
    device = torch.device(cfg.get("device", "cpu"))
    out_csv = os.path.join(cfg["results_dir"], "actor_baselines.csv")

    for seed in seeds:
        for name, params in defenses:
            t0 = time.time()
            row = run_one(
                "Actor", "GraphSAGE", name, params, int(seed), config=cfg, device=device
            )
            row["wall_seconds"] = round(time.time() - t0, 2)
            row["sami_config"] = "locked_cora_flagship" if name == "sami" else ""
            rows.append(row)
            print(
                f"Actor seed={seed} {name}: acc={row['test_accuracy']} "
                f"conf={row['conf_attack_auc']} lira={row['lira_attack_auc']} "
                f"tpr01={row.get('lira_tpr_at_0.01_fpr')} train_s={row.get('train_seconds')} "
                f"wall={row['wall_seconds']}s",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    cols = [
        "test_accuracy",
        "conf_attack_auc",
        "lira_attack_auc",
        "lira_tpr_at_0.01_fpr",
        "train_seconds",
    ]
    summary = df.groupby("defense")[cols].agg(["mean", "std"]).round(4)
    print(summary)
    means = df.groupby("defense")[cols].mean().round(4).to_dict()
    out = {
        "dataset": "Actor",
        "model": "GraphSAGE",
        "n_seeds": len(seeds),
        "n_shadows": 4,
        "locked_sami": LOCKED_SAMI,
        "means": means,
    }
    with open(os.path.join(cfg["results_dir"], "actor_baselines_summary.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
