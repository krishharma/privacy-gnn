"""
HARP at ogbn-arxiv scale: systems Mass/Frac/Acc/ECE (+ LiRA with n_shadows=2).

Demonstrates the *defense* (not only the undefended audit) at ~169k nodes.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import torch

from config import ensure_dirs, load_config
from defenses.harp import LOCKED_HARP
from experiment import run_one

SEEDS = [42, 123, 456]  # 3 seeds; Acc seed-bootstrap CIs in paper
LOCKED_SAMI = {
    "lam": 0.5,
    "use_lte": True,
    "use_gate": False,  # ogbn config uses gate false for speed
    "arch_aware": True,
    "noise_scale": 0.15,
    "budget_B": 0.0,
    "warmup_epochs": 3,
    "entropy_coef": 0.05,
}


def main():
    os.environ.setdefault("PRIVACYGNN_CONFIG", "experiment_config_ogbn.yaml")
    cfg = dict(load_config("experiment_config_ogbn.yaml"))
    ensure_dirs(cfg)
    cfg["lira"] = {"n_shadows": 2}
    cfg["attacks"] = ["confidence", "lira"]
    cfg["large_graph_use_official_split"] = True
    device = torch.device("cpu")

    # HARP at scale: slightly cheaper train alignment (gate off) but same Frac target
    harp = {
        **LOCKED_HARP,
        "use_gate": False,
        "warmup_epochs": 3,
        "strong_noise_scale": 0.30,
        "target_protect_frac": 0.40,
    }
    defenses = [
        ("none", {}),
        ("lbp", {"scale": 0.3}),
        ("harp", harp),
        ("sami", LOCKED_SAMI),
    ]
    rows = []
    out = "results/harp_ogbn.csv"
    t0 = time.time()
    for dn, dp in defenses:
        for seed in SEEDS:
            print(f"ogbn-arxiv/GraphSAGE/{dn} seed={seed}", flush=True)
            r = run_one("ogbn-arxiv", "GraphSAGE", dn, dp, seed, device=device, config=cfg)
            rows.append(r)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(
                f"  acc={r['test_accuracy']:.4f} lira={r['lira_attack_auc']:.4f} "
                f"mass={r.get('noise_mass')} frac={r.get('frac_protected')} "
                f"ece={r.get('ece_test')} train_s={r.get('train_seconds')}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    means = df.groupby("defense")[
        ["test_accuracy", "lira_attack_auc", "conf_attack_auc", "ece_test",
         "noise_mass", "frac_protected", "train_seconds"]
    ].mean()
    print(means, flush=True)
    summary = {
        "wall_s": round(time.time() - t0, 1),
        "n_shadows": 2,
        "seeds": SEEDS,
        "means": means.round(4).to_dict(),
    }
    with open("results/harp_ogbn_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("DONE", summary["wall_s"], "s", flush=True)


if __name__ == "__main__":
    main()
