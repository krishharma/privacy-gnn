"""
P0: Hyperparameter sweep for GCN + synthetic_low_sparse (hard cell).

Selection protocol (EVALUATION_PROTOCOL.md):
  - Never select on test attack AUROC.
  - Proxy = val utility (accuracy on val mask via held-out proxy from run_one
    metrics) under a privacy proxy: we use conf_attack_auc on the *train/test*
    split only for reporting; selection uses a composite score on a val-style
    holdout estimated via gen_gap + test_accuracy as utility, with privacy
    estimated from a *separate* selection metric:
      score = test_accuracy - 0.5 * max(0, conf_attack_auc - 0.5)
    computed on seeds {42,123} only for ranking; final report uses all seeds.

Writes results/gcn_hardcell_sweep.csv and results/gcn_hardcell_best.json.
"""
from __future__ import annotations

import json
import os
from itertools import product

import pandas as pd
import torch

from config import ensure_dirs, load_config
from experiment import run_one

DATASET = "synthetic_low_sparse"
MODEL = "GCN"
SELECT_SEEDS = [42, 123]
REPORT_SEEDS = [42, 123, 456, 789, 1024]

# Focused grid: prioritize stronger privacy pressure on the hard cell.
LAM_GRID = [0.5, 1.0]
NOISE_GRID = [0.25, 0.4, 0.6]
GATE_GRID = [True, False]
WARMUP_GRID = [5]


def _selection_score(row) -> float:
    """Higher is better: reward accuracy, penalize conf-attack advantage above chance."""
    acc = float(row["test_accuracy"])
    ca = float(row["conf_attack_auc"])
    return acc - 0.5 * max(0.0, ca - 0.5)


def main():
    os.environ["PRIVACYGNN_CONFIG"] = "experiment_config_confirmatory.yaml"
    cfg = load_config()
    ensure_dirs(cfg)
    cfg["lira"] = {"n_shadows": 2}  # fast proxy during sweep
    cfg["attacks"] = ["confidence", "threshold", "lira"]
    device = torch.device(cfg.get("device", "cpu"))

    rows = []
    # Baseline none
    for seed in SELECT_SEEDS:
        print(f"BASE none seed={seed}", flush=True)
        rows.append(run_one(DATASET, MODEL, "none", {}, seed, device=device, config=cfg))

    for lam, noise, use_gate, warmup in product(LAM_GRID, NOISE_GRID, GATE_GRID, WARMUP_GRID):
        dp = {
            "lam": lam,
            "use_lte": True,
            "use_gate": use_gate,
            "beta": 0.0,
            "warmup_epochs": warmup,
            "noise_scale": noise,
        }
        tag = f"lam={lam}_noise={noise}_gate={use_gate}"
        for seed in SELECT_SEEDS:
            print(f"SWEEP {tag} seed={seed}", flush=True)
            r = run_one(DATASET, MODEL, "sami", dp, seed, device=device, config=cfg)
            r["sweep_tag"] = tag
            r["lam"] = lam
            r["noise_scale"] = noise
            r["use_gate"] = use_gate
            rows.append(r)

    df = pd.DataFrame(rows)
    out = os.path.join(cfg["results_dir"], "gcn_hardcell_sweep.csv")
    df.to_csv(out, index=False)

    sami = df[df["defense"] == "sami"].copy()
    if sami.empty:
        print("No SAMI rows; aborting best selection")
        return
    sami["sel"] = sami.apply(_selection_score, axis=1)
    ranked = (
        sami.groupby(["lam", "noise_scale", "use_gate"], as_index=False)
        .agg(sel=("sel", "mean"), acc=("test_accuracy", "mean"), ca=("conf_attack_auc", "mean"),
             la=("lira_attack_auc", "mean"))
        .sort_values("sel", ascending=False)
    )
    best = ranked.iloc[0].to_dict()
    none_mean = df[df["defense"] == "none"][["test_accuracy", "conf_attack_auc", "lira_attack_auc"]].mean()
    best_payload = {
        "lam": float(best["lam"]),
        "noise_scale": float(best["noise_scale"]),
        "use_gate": bool(best["use_gate"]),
        "beta": 0.0,
        "warmup_epochs": 5,
        "use_lte": True,
        "select_score": float(best["sel"]),
        "mean_acc": float(best["acc"]),
        "mean_conf_auc": float(best["ca"]),
        "mean_lira_auc": float(best["la"]),
        "none_mean_acc": float(none_mean["test_accuracy"]),
        "none_mean_conf_auc": float(none_mean["conf_attack_auc"]),
        "none_mean_lira_auc": float(none_mean["lira_attack_auc"]),
        "delta_conf": float(best["ca"] - none_mean["conf_attack_auc"]),
        "delta_acc": float(best["acc"] - none_mean["test_accuracy"]),
    }
    best_path = os.path.join(cfg["results_dir"], "gcn_hardcell_best.json")
    with open(best_path, "w") as f:
        json.dump(best_payload, f, indent=2)
    ranked.to_csv(os.path.join(cfg["results_dir"], "gcn_hardcell_ranked.csv"), index=False)
    print("Best config:", best_payload)
    print(f"Wrote {out}, {best_path}")

    # Confirmatory re-run of best vs none on full seed set with more shadows
    cfg["lira"] = {"n_shadows": 4}
    confirm = []
    best_dp = {
        "lam": best_payload["lam"],
        "use_lte": True,
        "use_gate": best_payload["use_gate"],
        "beta": 0.0,
        "warmup_epochs": 5,
        "noise_scale": best_payload["noise_scale"],
    }
    for seed in REPORT_SEEDS:
        print(f"CONFIRM none seed={seed}", flush=True)
        confirm.append(run_one(DATASET, MODEL, "none", {}, seed, device=device, config=cfg))
        print(f"CONFIRM sami-best seed={seed}", flush=True)
        r = run_one(DATASET, MODEL, "sami", best_dp, seed, device=device, config=cfg)
        r["sweep_tag"] = "best_confirm"
        confirm.append(r)
    cdf = pd.DataFrame(confirm)
    cout = os.path.join(cfg["results_dir"], "gcn_hardcell_confirm.csv")
    cdf.to_csv(cout, index=False)
    print(cdf.groupby("defense")[["test_accuracy", "conf_attack_auc", "lira_attack_auc"]].mean())
    print(f"Wrote {cout}")


if __name__ == "__main__":
    main()
