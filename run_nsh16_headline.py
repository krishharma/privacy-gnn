#!/usr/bin/env python3
"""Headline Cora table at n_shadows=16 (modern LiRA budget), 5 seeds."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from config import load_config
from defenses.harp import LOCKED_HARP_RELEASE
from experiment import run_one

OUT = "results"
SEEDS5 = [42, 123, 456, 789, 1024]


def main():
    cfg = load_config("experiment_config_confirmatory.yaml")
    cfg["lira"] = {"n_shadows": 16}
    path = os.path.join(OUT, "harp_headline_nsh16.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.isfile(path) else []
    done = {(r["tag"], int(r["seed"])) for r in rows}
    jobs = [
        ("none", "none", {}),
        ("lbp_eq", "lbp", {"scale": 0.12}),
        ("lbp_strong", "lbp", {"scale": 0.30}),
        ("harp_locked", "harp_release_only", dict(LOCKED_HARP_RELEASE)),
        ("harp_full", "harp_uniform", {**LOCKED_HARP_RELEASE, "target_protect_frac": 1.0, "risk_frac": 1.0, "k_hops": 0}),
        ("gap_s3", "gap_agg", {"sigma": 3.0, "epochs": 80, "max_degree": 100}),
        ("memguard", "memguard", {"max_l1": 0.3, "n_steps": 60}),
    ]
    for seed in SEEDS5:
        for tag, dn, dp in jobs:
            if (tag, seed) in done:
                continue
            print(f"N16 {tag} seed={seed}", flush=True)
            r = run_one("Cora", "GraphSAGE", dn, dp, seed, config=cfg)
            rows.append({
                "tag": tag, "seed": seed,
                "Acc": float(r["test_accuracy"]),
                "LiRA": float(r["lira_attack_auc"]),
                "TPR1": float(r.get("lira_tpr_at_0.01_fpr", np.nan)),
                "ECE": float(r.get("ece_test", np.nan)),
                "ExactFrac": float(r.get("exact_frac", np.nan)),
                "phi_auc": float(r.get("mlp_phi_attack_auc", np.nan)),
                "conf_auc": float(r.get("conf_attack_auc", np.nan)),
                "eps": r.get("dp_epsilon"),
            })
            done.add((tag, seed))
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  Acc={rows[-1]['Acc']:.3f} LiRA={rows[-1]['LiRA']:.3f} phi={rows[-1]['phi_auc']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    means = df.groupby("tag").agg(
        Acc=("Acc", "mean"), LiRA=("LiRA", "mean"), TPR1=("TPR1", "mean"),
        ECE=("ECE", "mean"), ExactFrac=("ExactFrac", "mean"),
        phi=("phi_auc", "mean"), eps=("eps", "mean"),
        Acc_std=("Acc", "std"), LiRA_std=("LiRA", "std"), n=("Acc", "count"),
    ).reset_index()
    means.to_csv(os.path.join(OUT, "harp_headline_nsh16_means.csv"), index=False)
    print(means.to_string(), flush=True)
    print("N16 DONE", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PRIVACYGNN_CONFIG", "experiment_config_confirmatory.yaml")
    main()
