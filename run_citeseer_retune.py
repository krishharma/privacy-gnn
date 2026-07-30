"""
Citeseer GraphSAGE second-flagship attempt.
Pre-registered rule: select σ∈{0.25,0.35}, λ∈{0.5,1.0} by mean VAL Acc on
SELECT_SEEDS only (no test LiRA in selection). Confirm 5 seeds; report LiRA.
Also evaluate locked Cora config as a baseline row.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import torch

from config import ensure_dirs, load_config
from experiment import run_one

SELECT_SEEDS = [42, 123]
CONFIRM_SEEDS = [42, 123, 456, 789, 1024]
LOCKED_CORA = {
    "lam": 0.5,
    "use_lte": True,
    "use_gate": True,
    "arch_aware": True,
    "noise_scale": 0.35,
    "budget_B": 0.0,
    "warmup_epochs": 5,
    "entropy_coef": 0.05,
}
OUT_SEL = "results/citeseer_retune_select.csv"
OUT_CONF = "results/citeseer_retune_confirm.csv"
OUT_JSON = "results/citeseer_retune_summary.json"


def main():
    cfg = dict(load_config())
    ensure_dirs(cfg)
    cfg["attacks"] = ["confidence", "lira"]
    cfg["lira"] = {"n_shadows": 4}
    device = torch.device(cfg.get("device", "cpu"))

    cands = []
    for lam in (0.5, 1.0):
        for noise in (0.25, 0.35):
            cands.append(
                {
                    "lam": lam,
                    "use_lte": True,
                    "use_gate": True,
                    "arch_aware": True,
                    "noise_scale": noise,
                    "budget_B": 0.0,
                    "warmup_epochs": 5,
                    "entropy_coef": 0.05,
                }
            )

    # --- Selection on VAL Acc only ---
    sel_rows = []
    best, best_val = None, -1e9
    for params in cands:
        vals = []
        for seed in SELECT_SEEDS:
            print(f"select lam={params['lam']} σ={params['noise_scale']} seed={seed}", flush=True)
            r = run_one("Citeseer", "GraphSAGE", "sami", params, seed, config=cfg, device=device)
            r["phase"] = "select"
            r["cand"] = json.dumps(params)
            sel_rows.append(r)
            vals.append(float(r.get("val_accuracy", r["test_accuracy"])))
        m = sum(vals) / len(vals)
        print(f"  mean_val_acc={m:.4f}", flush=True)
        if m > best_val:
            best_val = m
            best = params
    pd.DataFrame(sel_rows).to_csv(OUT_SEL, index=False)

    baselines = [
        ("none", {}),
        ("gtd", {"gamma": 1.0, "stage1_frac": 0.5, "pseudo_conf": 0.8}),
        ("lbp", {"scale": 0.3}),
        ("maskarmor", {"top_k": 1}),
        ("sami_locked_cora", LOCKED_CORA),
        ("sami_selected", best),
    ]

    conf = []
    for seed in CONFIRM_SEEDS:
        for tag, params in baselines:
            name = "sami" if tag.startswith("sami") else tag
            print(f"confirm {tag} seed={seed}", flush=True)
            t0 = time.time()
            r = run_one("Citeseer", "GraphSAGE", name, params, seed, config=cfg, device=device)
            r["variant"] = tag
            r["phase"] = "confirm"
            r["wall_seconds"] = round(time.time() - t0, 2)
            conf.append(r)
            pd.DataFrame(conf).to_csv(OUT_CONF, index=False)

    df = pd.DataFrame(conf)
    means = (
        df.groupby("variant")[
            ["test_accuracy", "val_accuracy", "conf_attack_auc", "lira_attack_auc", "lira_tpr_at_0.01_fpr"]
        ]
        .mean()
        .round(4)
    )
    none_lira = float(means.loc["none", "lira_attack_auc"])
    none_acc = float(means.loc["none", "test_accuracy"])
    sel_lira = float(means.loc["sami_selected", "lira_attack_auc"])
    sel_acc = float(means.loc["sami_selected", "test_accuracy"])
    win = (none_lira - sel_lira) >= 0.03 and (none_acc - sel_acc) <= 0.01
    summary = {
        "selected_params": best,
        "select_mean_val_acc": round(best_val, 4),
        "means": means.to_dict(),
        "delta_lira_selected_vs_none": round(sel_lira - none_lira, 4),
        "delta_acc_selected_vs_none": round(sel_acc - none_acc, 4),
        "second_flagship_bar_met": bool(win),
        "bar": "LiRA drop >=0.03 with Acc within 0.01 of none",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(means)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
