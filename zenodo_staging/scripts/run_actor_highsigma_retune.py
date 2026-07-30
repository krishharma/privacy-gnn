"""
Actor higher-σ val-only retune vs LBP (pre-registered).

Selection rule (no test LiRA):
  1) On SELECT seeds, measure LBP mean val Acc.
  2) Grid λ∈{0.5,1.0} × σ∈{0.35,0.4,0.5,0.6}.
  3) Feasible = mean val Acc >= LBP_val + 0.03.
  4) Among feasible, pick highest σ then highest λ (privacy-preferring tiebreak).
  5) If none feasible, pick max val Acc (report failure to meet Acc floor).

Confirm 5 seeds: none / LBP / sami_selected / sami_locked035.
Joint win: Acc >= LBP+0.03 and LiRA <= LBP+0.005.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import torch

from config import ensure_dirs, load_config
from experiment import run_one

SELECT = [42, 123]
CONFIRM = [42, 123, 456, 789, 1024]
BASE = {
    "use_lte": True,
    "use_gate": True,
    "arch_aware": True,
    "budget_B": 0.0,
    "warmup_epochs": 5,
    "entropy_coef": 0.05,
}
OUT = "results/actor_highsigma_retune.csv"
OUT_JSON = "results/actor_highsigma_retune_summary.json"


def main():
    cfg = dict(load_config())
    ensure_dirs(cfg)
    cfg["attacks"] = ["confidence", "lira"]
    cfg["lira"] = {"n_shadows": 4}
    device = torch.device("cpu")

    rows = []
    # LBP val Acc floor
    lbp_vals = []
    for seed in SELECT:
        print(f"select LBP seed={seed}", flush=True)
        r = run_one("Actor", "GraphSAGE", "lbp", {"scale": 0.3}, seed, config=cfg, device=device)
        r["phase"] = "select"
        r["variant"] = "lbp"
        rows.append(r)
        lbp_vals.append(float(r["val_accuracy"]))
    lbp_val = sum(lbp_vals) / len(lbp_vals)
    floor = lbp_val + 0.03
    print(f"LBP mean val Acc={lbp_val:.4f}; Acc floor={floor:.4f}", flush=True)

    cands = []
    for lam in (0.5, 1.0):
        for sigma in (0.35, 0.4, 0.5, 0.6):
            cands.append((lam, sigma))

    scored = []
    for lam, sigma in cands:
        params = dict(BASE, lam=lam, noise_scale=sigma)
        vals = []
        for seed in SELECT:
            print(f"select λ={lam} σ={sigma} seed={seed}", flush=True)
            r = run_one("Actor", "GraphSAGE", "sami", params, seed, config=cfg, device=device)
            r["phase"] = "select"
            r["variant"] = f"sami_lam{lam}_sig{sigma}"
            r["lam"] = lam
            r["sigma"] = sigma
            rows.append(r)
            vals.append(float(r["val_accuracy"]))
            pd.DataFrame(rows).to_csv(OUT, index=False)
        m = sum(vals) / len(vals)
        feasible = m >= floor
        scored.append({"lam": lam, "sigma": sigma, "val_acc": m, "feasible": feasible})
        print(f"  mean_val={m:.4f} feasible={feasible}", flush=True)

    feasible = [s for s in scored if s["feasible"]]
    if feasible:
        # highest σ, then highest λ
        best = sorted(feasible, key=lambda s: (s["sigma"], s["lam"]), reverse=True)[0]
        rule = "feasible_max_sigma"
    else:
        best = sorted(scored, key=lambda s: s["val_acc"], reverse=True)[0]
        rule = "fallback_max_val_acc"
    best_params = dict(BASE, lam=best["lam"], noise_scale=best["sigma"])
    print(f"SELECTED rule={rule} {best_params}", flush=True)

    for seed in CONFIRM:
        for name, p, tag in [
            ("none", {}, "none"),
            ("lbp", {"scale": 0.3}, "lbp"),
            ("sami", best_params, "sami_selected"),
            ("sami", dict(BASE, lam=0.5, noise_scale=0.35), "sami_locked035"),
        ]:
            print(f"confirm {tag} seed={seed}", flush=True)
            r = run_one("Actor", "GraphSAGE", name, p, seed, config=cfg, device=device)
            r["phase"] = "confirm"
            r["variant"] = tag
            rows.append(r)
            pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    conf = df[df.phase == "confirm"]
    means = (
        conf.groupby("variant")[["test_accuracy", "lira_attack_auc", "conf_attack_auc", "val_accuracy"]]
        .mean()
        .round(4)
    )
    beats = bool(
        means.loc["sami_selected", "lira_attack_auc"]
        <= means.loc["lbp", "lira_attack_auc"] + 0.005
        and means.loc["sami_selected", "test_accuracy"]
        >= means.loc["lbp", "test_accuracy"] + 0.03
    )
    summary = {
        "selection_rule": rule,
        "lbp_mean_val_acc": round(lbp_val, 4),
        "acc_floor": round(floor, 4),
        "select_scores": scored,
        "selected_params": best_params,
        "means": means.to_dict(),
        "beats_lbp_joint": beats,
        "delta_lira_vs_lbp": round(
            float(means.loc["sami_selected", "lira_attack_auc"] - means.loc["lbp", "lira_attack_auc"]), 4
        ),
        "delta_acc_vs_lbp": round(
            float(means.loc["sami_selected", "test_accuracy"] - means.loc["lbp", "test_accuracy"]), 4
        ),
    }
    json.dump(summary, open(OUT_JSON, "w"), indent=2)
    print(means)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
