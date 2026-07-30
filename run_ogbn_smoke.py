"""
ogbn-arxiv smoke gate: GraphSAGE × none × seed 42 × n_shadows=1.
Writes results/ogbn_smoke_timing.json and sizes the Volume grid.
"""
from __future__ import annotations

import json
import os
import resource
import time
import tracemalloc

import numpy as np
import torch

from config import ensure_dirs, load_config
from experiment import run_one


def peak_rss_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 10**9:  # bytes (macOS)
        return rss / (1024 * 1024)
    return rss / 1024.0  # KB → MB


def main():
    os.environ["PRIVACYGNN_CONFIG"] = os.environ.get(
        "PRIVACYGNN_CONFIG", "experiment_config_ogbn_smoke.yaml"
    )
    cfg = load_config()
    ensure_dirs(cfg)
    device = torch.device(cfg.get("device", "cpu"))
    print(f"device={device} cuda={torch.cuda.is_available()}")

    tracemalloc.start()
    t0 = time.time()
    row = run_one(
        dataset_name="ogbn-arxiv",
        model_name="GraphSAGE",
        defense_name="none",
        defense_params={},
        seed=42,
        config=cfg,
        device=device,
    )
    wall = time.time() - t0
    _, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    train_s = float(row.get("train_seconds", float("nan")))
    # One cell = target + 1 shadow under this smoke config
    n_defenses = 4  # none, sami, gtd, lbp
    n_seeds = 3
    for n_shadows in (1, 2, 4):
        # Approximate: each seed trains 1 target + n_shadows shadows
        proj_h = (wall * n_defenses * n_seeds * (1 + n_shadows) / 2.0) / 3600.0
        # wall already includes 1 target + 1 shadow → scale by (1+n_s)/(1+1)
        proj_h = (wall * (1 + n_shadows) / 2.0 * n_defenses * n_seeds) / 3600.0

    projections = {}
    for n_shadows in (1, 2, 4):
        projections[f"n_shadows_{n_shadows}"] = {
            "hours_est_4def_3seeds": round(
                (wall * (1 + n_shadows) / 2.0 * n_defenses * n_seeds) / 3600.0, 3
            ),
            "hours_est_4def_5seeds": round(
                (wall * (1 + n_shadows) / 2.0 * n_defenses * 5) / 3600.0, 3
            ),
        }

    # Policy: prefer n_shadows that keeps 4def×3seeds under ~12h on CPU
    chosen_shadows = 2
    chosen_seeds = 3
    for ns in (4, 2, 1):
        h = projections[f"n_shadows_{ns}"]["hours_est_4def_3seeds"]
        if h <= 12.0:
            chosen_shadows = ns
            break
    if projections[f"n_shadows_{chosen_shadows}"]["hours_est_4def_5seeds"] <= 18.0:
        chosen_seeds = 5

    out = {
        "dataset": "ogbn-arxiv",
        "model": "GraphSAGE",
        "defense": "none",
        "seed": 42,
        "n_shadows_smoke": 1,
        "device": str(device),
        "wall_seconds_target_plus_1shadow": round(wall, 2),
        "train_seconds_target": train_s,
        "test_accuracy": float(row.get("test_accuracy", float("nan"))),
        "conf_attack_auc": float(row.get("conf_attack_auc", float("nan"))),
        "lira_attack_auc": float(row.get("lira_attack_auc", float("nan"))),
        "peak_python_alloc_mb": round(peak_py / (1024 * 1024), 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "projections": projections,
        "policy": {
            "n_shadows": chosen_shadows,
            "n_seeds": chosen_seeds,
            "defenses": ["none", "sami", "gtd", "lbp"],
            "maskarmor": False,
            "note": "Sized so confirmatory Volume grid stays within ~12–18h CPU budget.",
        },
    }
    path = os.path.join(cfg["results_dir"], "ogbn_smoke_timing.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
