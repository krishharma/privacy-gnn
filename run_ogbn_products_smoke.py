"""
ogbn-products memory-light smoke after OOM on n_shadows=2×3seeds.
Policy: 1 seed, n_shadows=1, none+SAMI, fewer epochs; Acc+conf+LiRA.
"""
from __future__ import annotations

import json
import os
import time
import traceback

import pandas as pd
import torch

from config import ensure_dirs, load_config
from experiment import run_one
from ogb_loader import load_large_benchmark

SAMI = {
    "lam": 0.5,
    "use_lte": True,
    "use_gate": False,
    "arch_aware": True,
    "warmup_epochs": 2,
    "entropy_coef": 0.05,
    "noise_scale": 0.1,
    "budget_B": 0.0,
}
OUT_CSV = "results/ogbn_products_smoke.csv"
OUT_JSON = "results/ogbn_products_smoke_summary.json"
PROBE = "results/ogbn_products_probe.json"


def main():
    os.environ["PRIVACYGNN_CONFIG"] = "experiment_config_ogbn.yaml"
    cfg = dict(load_config())
    ensure_dirs(cfg)
    root = os.path.join(cfg["data_dir"], "ogb")

    # Ensure loaded (already on disk)
    import builtins
    import ogb.utils.url as ogb_url

    ogb_url.decide_download = lambda url: True
    _input = builtins.input
    builtins.input = lambda *a, **k: "y"
    try:
        data, nc, nf = load_large_benchmark("ogbn-products", root)
    finally:
        builtins.input = _input

    probe = {
        "attempted": "ogbn-products",
        "status": "smoke_running",
        "n_nodes": int(data.num_nodes),
        "prior_full_grid": "OOM exit 137 at none seed=42 n_shadows=2 (~57min)",
        "smoke_policy": "1 seed, n_shadows=1, epochs=10, none+SAMI",
    }
    json.dump(probe, open(PROBE, "w"), indent=2)
    print(json.dumps(probe, indent=2), flush=True)

    cfg["lira"] = {"n_shadows": 1}
    cfg["attacks"] = ["confidence", "lira"]
    cfg["large_graph_use_official_split"] = True
    cfg["training"] = dict(cfg.get("training", {}), epochs=10)
    # smaller neighbor fanout / batch if present
    mb = dict(cfg.get("minibatch", {}))
    mb["batch_size"] = min(int(mb.get("batch_size", 1024)), 512)
    mb["num_neighbors"] = [10, 10]
    cfg["minibatch"] = mb
    device = torch.device("cpu")

    rows = []
    if os.path.isfile(OUT_CSV):
        rows = pd.read_csv(OUT_CSV).to_dict("records")
        done = {(int(r["seed"]), r["defense"]) for r in rows}
    else:
        done = set()

    for seed in [42]:
        for name, params in [("none", {}), ("sami", SAMI)]:
            if (seed, name) in done:
                continue
            print(f"SMOKE products seed={seed} {name}", flush=True)
            t0 = time.time()
            try:
                row = run_one(
                    "ogbn-products", "GraphSAGE", name, params, seed, config=cfg, device=device
                )
            except Exception as e:
                err = {"error": str(e), "traceback": traceback.format_exc(), "seed": seed, "defense": name}
                json.dump(err, open("results/ogbn_products_run_error.json", "w"), indent=2)
                probe["status"] = "smoke_failed"
                probe["error"] = str(e)
                json.dump(probe, open(PROBE, "w"), indent=2)
                raise
            row["wall_seconds"] = round(time.time() - t0, 2)
            row["n_shadows_run"] = 1
            row["note"] = "memory_light_smoke"
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            print(
                seed, name,
                "acc", row["test_accuracy"],
                "conf", row["conf_attack_auc"],
                "lira", row["lira_attack_auc"],
                "wall", row["wall_seconds"],
                flush=True,
            )

    df = pd.DataFrame(rows)
    summary = {
        "dataset": "ogbn-products",
        "mode": "memory_light_smoke",
        "n_shadows": 1,
        "seeds": [42],
        "epochs": 10,
        "n_nodes": int(data.num_nodes),
        "means": df.groupby("defense")[
            ["test_accuracy", "conf_attack_auc", "lira_attack_auc", "wall_seconds"]
        ]
        .mean()
        .round(4)
        .to_dict(),
        "note": "Full 3-seed n_shadows=2 OOM'd (exit 137). Smoke is systems+privacy sanity on 2.45M nodes.",
    }
    json.dump(summary, open(OUT_JSON, "w"), indent=2)
    probe["status"] = "smoke_complete"
    probe["summary"] = summary
    json.dump(probe, open(PROBE, "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
