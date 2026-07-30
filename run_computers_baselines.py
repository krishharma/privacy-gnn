"""
Amazon Computers (~13.7k nodes): second multi-k real graph baseline grid.
Fallback Volume×Variety bridge when ogbn-products cannot download (disk).
none / GTD / LBP / MaskArmor / locked SAMI, 5 seeds, n_shadows=4.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import torch
from torch_geometric.datasets import Amazon

from config import ensure_dirs, load_config
from data import resplit, homophily, density
from experiment import run_one

SEEDS = [42, 123, 456, 789, 1024]
LOCKED = {
    "lam": 0.5,
    "use_lte": True,
    "use_gate": True,
    "arch_aware": True,
    "noise_scale": 0.35,
    "budget_B": 0.0,
    "warmup_epochs": 5,
    "entropy_coef": 0.05,
}
OUT = "results/computers_baselines.csv"
OUT_JSON = "results/computers_baselines_summary.json"


def ensure_wired():
    """Register Amazon Computers into experiment load path via monkeypatch if needed."""
    # Prefer native: add to experiment by using dataset name after patching _load_target_data
    pass


def main():
    cfg = dict(load_config())
    ensure_dirs(cfg)
    cfg["attacks"] = ["confidence", "lira"]
    cfg["lira"] = {"n_shadows": 4}
    device = torch.device("cpu")

    # Probe size
    ds = Amazon(root=os.path.join(cfg["data_dir"], "Computers"), name="Computers")
    data0 = ds[0]
    meta = {
        "n_nodes": int(data0.num_nodes),
        "n_edges": int(data0.edge_index.size(1)),
        "num_classes": int(ds.num_classes),
        "num_features": int(ds.num_features),
    }
    print(meta, flush=True)

    # Wire into experiment.py loaders temporarily
    import experiment as exp
    import data as data_mod

    def load_computers(name, data_dir=None):
        d = Amazon(root=os.path.join(data_dir or cfg["data_dir"], "Computers"), name="Computers")
        return d[0], d.num_classes, d.num_features

    _orig_target = exp._load_target_data
    _orig_shadow = exp._make_shadow_data

    def _load_target(dataset_name, data_dir, seed, use_official_large, split_kw):
        if dataset_name == "Computers":
            data, nc, nf = load_computers(dataset_name, data_dir)
            ratios = exp._resplit_kwargs(split_kw)
            return resplit(data, seed, **ratios), nc, nf
        return _orig_target(dataset_name, data_dir, seed, use_official_large, split_kw)

    def _load_shadow(dataset_name, data_dir, shadow_seed, split_kw):
        if dataset_name == "Computers":
            data, nc, nf = load_computers(dataset_name, data_dir)
            ratios = exp._resplit_kwargs(split_kw)
            return resplit(data, shadow_seed, **ratios), nc, nf
        return _orig_shadow(dataset_name, data_dir, shadow_seed, split_kw)

    exp._load_target_data = _load_target
    exp._make_shadow_data = _load_shadow

    defenses = [
        ("none", {}),
        ("gtd", {"gamma": 1.0, "stage1_frac": 0.5, "pseudo_conf": 0.8}),
        ("lbp", {"scale": 0.3}),
        ("maskarmor", {"top_k": 1}),
        ("sami", LOCKED),
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
            print(f"Computers seed={seed} {name}", flush=True)
            t0 = time.time()
            row = run_one("Computers", "GraphSAGE", name, params, seed, config=cfg, device=device)
            row["wall_seconds"] = round(time.time() - t0, 2)
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(seed, name, row["test_accuracy"], row["lira_attack_auc"], row["conf_attack_auc"], flush=True)

    df = pd.DataFrame(rows)
    means = (
        df.groupby("defense")[
            ["test_accuracy", "conf_attack_auc", "lira_attack_auc", "lira_tpr_at_0.01_fpr", "train_seconds"]
        ]
        .mean()
        .round(4)
    )
    summary = {
        "dataset": "Computers",
        "meta": meta,
        "homophily": float(df["homophily"].mean()) if "homophily" in df.columns else None,
        "means": means.to_dict(),
        "note": "Amazon Computers multi-k grid; Volume×Variety bridge when products unavailable.",
    }
    json.dump(summary, open(OUT_JSON, "w"), indent=2)
    print(means)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
