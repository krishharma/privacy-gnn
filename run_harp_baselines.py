"""
HARP primary evaluation grid (CPU-friendly).

Datasets: Cora, Citeseer, Chameleon, Actor, Photo, PubMed
Model: GraphSAGE (primary); Cora also runs GCN for a hard-cell check
Defenses: none, lbp, gtd, sami (locked), harp (locked), harp_k0, harp_k2,
          harp_uniform, harp_release_only
Seeds: 5 (citation/hetero); Photo/PubMed: 3
n_shadows: 4

Writes:
  results/harp_baselines.csv
  results/harp_baselines_summary.json
  results/harp_means.csv
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

PRIMARY_DATASETS = ["Cora", "Citeseer", "Chameleon", "Actor"]
CONTROL_DATASETS = ["PubMed"]
SEEDS_PRIMARY = [42, 123, 456]
SEEDS_CONTROL = [42, 123, 456]

DEFENSES = [
    ("none", {}),
    ("lbp", {"scale": 0.3}),
    ("gtd", {"gamma": 1.0, "stage1_frac": 0.5, "pseudo_conf": 0.8}),
    ("sami", LOCKED_SAMI),
    ("harp", dict(LOCKED_HARP)),
    ("harp_k0", {**LOCKED_HARP, "k_hops": 0}),
    ("harp_k2", {**LOCKED_HARP, "k_hops": 2}),
    ("harp_uniform", {
        **LOCKED_HARP,
        "risk_frac": 1.0,
        "k_hops": 0,
        "use_gate": False,
        "train_on_protected": False,
    }),
    ("harp_release_only", {
        **LOCKED_HARP,
        "lam": 0.0,
        "use_gate": False,
        "train_on_protected": False,
    }),
]


def _summarize(df: pd.DataFrame) -> dict:
    keys = [
        "test_accuracy",
        "lira_attack_auc",
        "conf_attack_auc",
        "noise_mass",
        "frac_protected",
        "frac_seeds",
        "mean_scale",
        "relative_noise_mass_vs_uniform",
        "train_seconds",
        "lira_tpr_at_0.01_fpr",
    ]
    out = {}
    for (ds, model, defense), g in df.groupby(["dataset", "model", "defense"]):
        row = {"n": int(len(g))}
        for k in keys:
            if k in g.columns:
                row[k] = float(g[k].mean())
                row[f"{k}_std"] = float(g[k].std(ddof=0)) if len(g) > 1 else 0.0
        out[f"{ds}/{model}/{defense}"] = row
    return out


def main():
    os.environ.setdefault("PRIVACYGNN_CONFIG", "experiment_config_confirmatory.yaml")
    cfg = load_config()
    ensure_dirs(cfg)
    cfg = dict(cfg)
    cfg["lira"] = {"n_shadows": 4}
    cfg["attacks"] = ["confidence", "lira", "threshold"]
    device = torch.device(cfg.get("device", "cpu"))

    rows = []
    t0 = time.time()
    out_csv = os.path.join(cfg["results_dir"], "harp_baselines.csv")

    def _flush():
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)

    # Primary GraphSAGE grid
    for ds in PRIMARY_DATASETS:
        for dn, dp in DEFENSES:
            for seed in SEEDS_PRIMARY:
                print(f"{ds}/GraphSAGE/{dn} seed={seed}", flush=True)
                rows.append(
                    run_one(ds, "GraphSAGE", dn, dp, seed, device=device, config=cfg)
                )
                _flush()

    # Cora GCN hard-cell (none + harp + sami + lbp)
    for dn, dp in [
        ("none", {}),
        ("lbp", {"scale": 0.3}),
        ("sami", LOCKED_SAMI),
        ("harp", dict(LOCKED_HARP)),
    ]:
        for seed in SEEDS_PRIMARY:
            print(f"Cora/GCN/{dn} seed={seed}", flush=True)
            rows.append(run_one("Cora", "GCN", dn, dp, seed, device=device, config=cfg))
            _flush()

    # Near-chance controls
    for ds in CONTROL_DATASETS:
        for dn, dp in [
            ("none", {}),
            ("lbp", {"scale": 0.3}),
            ("sami", LOCKED_SAMI),
            ("harp", dict(LOCKED_HARP)),
        ]:
            for seed in SEEDS_CONTROL:
                print(f"{ds}/GraphSAGE/{dn} seed={seed}", flush=True)
                rows.append(
                    run_one(ds, "GraphSAGE", dn, dp, seed, device=device, config=cfg)
                )
                _flush()

    df = pd.DataFrame(rows)

    summary = _summarize(df)
    summary["_meta"] = {
        "elapsed_sec": round(time.time() - t0, 1),
        "locked_harp": LOCKED_HARP,
        "n_rows": int(len(df)),
    }
    out_json = os.path.join(cfg["results_dir"], "harp_baselines_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Compact means table for paper
    mean_rows = []
    for key, vals in summary.items():
        if key.startswith("_"):
            continue
        ds, model, defense = key.split("/")
        mean_rows.append({"dataset": ds, "model": model, "defense": defense, **{
            k: vals[k] for k in vals if not k.endswith("_std") and k != "n"
        }, "n_seeds": vals["n"]})
    means = pd.DataFrame(mean_rows)
    means_path = os.path.join(cfg["results_dir"], "harp_means.csv")
    means.to_csv(means_path, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")
    print(f"Wrote {out_json}")
    print(f"Wrote {means_path}")
    print(means.to_string(index=False))


if __name__ == "__main__":
    main()
