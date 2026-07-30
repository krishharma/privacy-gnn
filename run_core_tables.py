"""
Run a compact confirmatory core for paper tables (fast path).
Writes results/core_results.csv then regenerates BigData figures from
all_results.csv if present (falls back to core_results).
"""
import os
import pandas as pd
import torch
from config import load_config, ensure_dirs
from experiment import run_one

CORE = []
DATASETS = [
    "Cora",
    "Citeseer",
    "PubMed",
    "synthetic_low_sparse",
    "synthetic_low_medium",
    "synthetic_high_dense",
]
MODELS_GNN = ["GCN", "GraphSAGE"]
MODELS_BASE = ["LogReg", "MLP"]
SEEDS = [42, 123, 456, 789, 1024]
DEFENSES = [
    ("none", {}),
    ("dropedge", {"rate": 0.3}),
    ("label_smoothing", {"alpha": 0.1}),
    ("lbp", {"scale": 0.3}),
    ("gtd", {"gamma": 1.0, "stage1_frac": 0.5, "pseudo_conf": 0.8}),
    ("sami", {"lam": 0.5, "use_lte": True, "use_gate": True, "beta": 0.0, "warmup_epochs": 5, "noise_scale": 0.35}),
]
ABLATIONS = [
    ("sami_no_lte", {"lam": 0.5, "use_lte": False, "use_gate": True, "beta": 0.0, "warmup_epochs": 5, "noise_scale": 0.35}),
    ("sami_no_adv", {"lam": 0.0, "use_lte": True, "use_gate": True, "beta": 0.0, "warmup_epochs": 5, "noise_scale": 0.35}),
    ("sami_no_gate", {"lam": 0.5, "use_lte": True, "use_gate": False, "beta": 0.0, "warmup_epochs": 5, "noise_scale": 0.35}),
    ("sami_temp_only", {"lam": 0.0, "use_lte": True, "use_gate": False, "beta": 1.5, "warmup_epochs": 5, "noise_scale": 0.0}),
    ("advreg", {"lam": 0.5, "use_lte": False, "use_gate": False, "beta": 0.0, "warmup_epochs": 5, "noise_scale": 0.0}),
]
ABLATION_DATASETS = ["Cora", "synthetic_low_sparse"]


def main():
    # Use confirmatory config for split/lira/attack settings, but our own grid.
    os.environ["PRIVACYGNN_CONFIG"] = "experiment_config_confirmatory.yaml"
    cfg = load_config()
    ensure_dirs(cfg)
    cfg["lira"] = {"n_shadows": 4}
    device = torch.device(cfg.get("device", "cpu"))
    results = []

    # Discovery + main defenses
    for ds in DATASETS:
        for model in MODELS_BASE:
            for seed in SEEDS:
                results.append(
                    run_one(ds, model, "none", {}, seed, device=device, config=cfg)
                )
        for model in MODELS_GNN:
            for dn, dp in DEFENSES:
                for seed in SEEDS:
                    print(f"{ds}/{model}/{dn} seed={seed}", flush=True)
                    results.append(
                        run_one(ds, model, dn, dp, seed, device=device, config=cfg)
                    )

    # Ablations on core datasets
    for ds in ABLATION_DATASETS:
        for model in MODELS_GNN:
            for dn, dp in ABLATIONS:
                for seed in SEEDS:
                    print(f"ABL {ds}/{model}/{dn} seed={seed}", flush=True)
                    results.append(
                        run_one(ds, model, dn, dp, seed, device=device, config=cfg)
                    )

    out = os.path.join(cfg["results_dir"], "core_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(out, index=False)
    # Also merge into all_results for figure scripts if confirmatory still running.
    all_path = os.path.join(cfg["results_dir"], "all_results.csv")
    df.to_csv(all_path, index=False)
    print(f"Wrote {len(df)} rows to {out} and {all_path}")


if __name__ == "__main__":
    main()
