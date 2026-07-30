"""
Actor: val-only σ∈{0.25,0.35} selection (SELECT_SEEDS), confirm vs LBP on joint frontier.
Pre-registered: pick σ by mean val Acc; report Acc/LiRA vs LBP.
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
    "lam": 0.5,
    "use_lte": True,
    "use_gate": True,
    "arch_aware": True,
    "budget_B": 0.0,
    "warmup_epochs": 5,
    "entropy_coef": 0.05,
}
OUT = "results/actor_sigma_retune.csv"
OUT_JSON = "results/actor_sigma_retune_summary.json"


def main():
    cfg = dict(load_config())
    ensure_dirs(cfg)
    cfg["attacks"] = ["confidence", "lira"]
    cfg["lira"] = {"n_shadows": 4}
    device = torch.device("cpu")

    best_sigma, best_val = None, -1e9
    rows = []
    for sigma in (0.25, 0.35):
        params = dict(BASE, noise_scale=sigma)
        vals = []
        for seed in SELECT:
            print(f"select σ={sigma} seed={seed}", flush=True)
            r = run_one("Actor", "GraphSAGE", "sami", params, seed, config=cfg, device=device)
            r["phase"] = "select"
            r["sigma"] = sigma
            rows.append(r)
            vals.append(float(r["val_accuracy"]))
        m = sum(vals) / len(vals)
        print(f"σ={sigma} mean_val={m:.4f}", flush=True)
        if m > best_val:
            best_val, best_sigma = m, sigma

    params = dict(BASE, noise_scale=best_sigma)
    for seed in CONFIRM:
        for name, p, tag in [
            ("none", {}, "none"),
            ("lbp", {"scale": 0.3}, "lbp"),
            ("sami", params, "sami_selected"),
            ("sami", dict(BASE, noise_scale=0.35), "sami_locked035"),
        ]:
            print(f"confirm {tag} seed={seed}", flush=True)
            r = run_one("Actor", "GraphSAGE", name, p, seed, config=cfg, device=device)
            r["phase"] = "confirm"
            r["variant"] = tag
            rows.append(r)
            pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    conf = df[df.phase == "confirm"]
    means = conf.groupby("variant")[["test_accuracy", "lira_attack_auc", "conf_attack_auc"]].mean().round(4)
    summary = {
        "selected_sigma": best_sigma,
        "select_mean_val": round(best_val, 4),
        "means": means.to_dict(),
        "beats_lbp_joint": bool(
            means.loc["sami_selected", "lira_attack_auc"]
            <= means.loc["lbp", "lira_attack_auc"] + 0.005
            and means.loc["sami_selected", "test_accuracy"]
            >= means.loc["lbp", "test_accuracy"] + 0.03
        ),
    }
    json.dump(summary, open(OUT_JSON, "w"), indent=2)
    print(means)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
