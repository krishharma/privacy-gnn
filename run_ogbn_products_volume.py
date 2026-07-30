"""
ogbn-products Volume audit: download if needed, then none+SAMI with LiRA.
Starts with n_shadows=2, 3 seeds (same Volume policy as arxiv main grid).
Writes incremental CSV; resumes if interrupted.
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

SEEDS = [42, 123, 456]
SAMI = {
    "lam": 0.5,
    "use_lte": True,
    "use_gate": False,
    "arch_aware": True,
    "warmup_epochs": 3,
    "entropy_coef": 0.05,
    "noise_scale": 0.1,
    "budget_B": 0.0,
}
OUT_CSV = "results/ogbn_products_volume.csv"
OUT_JSON = "results/ogbn_products_volume_summary.json"
PROBE = "results/ogbn_products_probe.json"


def main():
    os.environ["PRIVACYGNN_CONFIG"] = "experiment_config_ogbn.yaml"
    cfg = dict(load_config())
    ensure_dirs(cfg)
    root = os.path.join(cfg["data_dir"], "ogb")
    os.makedirs(root, exist_ok=True)

    probe = {"attempted": "ogbn-products", "status": "starting", "root": root}
    json.dump(probe, open(PROBE, "w"), indent=2)

    print("Loading/downloading ogbn-products (may be multi-GB)...", flush=True)
    t0 = time.time()
    try:
        # OGB prompts interactively (download + update); auto-accept.
        import builtins
        import ogb.utils.url as ogb_url

        ogb_url.decide_download = lambda url: True
        _input = builtins.input
        builtins.input = lambda *a, **k: "y"
        try:
            # Clear broken empty stub that triggers "update?" prompt
            stub = os.path.join(root, "ogbn_products")
            if os.path.isdir(stub) and not os.path.isfile(
                os.path.join(stub, "processed", "geometric_data_processed.pt")
            ):
                import shutil

                shutil.rmtree(stub, ignore_errors=True)
            data, nc, nf = load_large_benchmark("ogbn-products", root)
        finally:
            builtins.input = _input
        probe.update(
            {
                "status": "loaded",
                "n_nodes": int(data.num_nodes),
                "n_edges": int(data.edge_index.size(1)),
                "num_classes": int(nc),
                "num_features": int(nf),
                "load_wall_s": round(time.time() - t0, 1),
                "train_n": int(data.train_mask.sum()),
                "val_n": int(data.val_mask.sum()),
                "test_n": int(data.test_mask.sum()),
            }
        )
        json.dump(probe, open(PROBE, "w"), indent=2)
        print(json.dumps(probe, indent=2), flush=True)
    except Exception as e:
        probe.update({"status": "download_or_load_failed", "error": str(e), "traceback": traceback.format_exc()})
        json.dump(probe, open(PROBE, "w"), indent=2)
        print(probe, flush=True)
        raise

    cfg["lira"] = {"n_shadows": 2}
    cfg["attacks"] = ["confidence", "lira"]
    cfg["large_graph_use_official_split"] = True
    # products is huge — keep epochs from ogbn yaml (20)
    device = torch.device(cfg.get("device", "cpu"))

    rows = []
    if os.path.isfile(OUT_CSV):
        rows = pd.read_csv(OUT_CSV).to_dict("records")
        done = {(int(r["seed"]), r["defense"]) for r in rows}
        print(f"resume {len(done)} done", flush=True)
    else:
        done = set()

    for seed in SEEDS:
        for name, params in [("none", {}), ("sami", SAMI)]:
            if (seed, name) in done:
                print(f"skip {seed} {name}", flush=True)
                continue
            print(f"RUN products seed={seed} {name} n_shadows=2", flush=True)
            tw = time.time()
            try:
                row = run_one(
                    "ogbn-products",
                    "GraphSAGE",
                    name,
                    params,
                    seed,
                    config=cfg,
                    device=device,
                )
            except Exception as e:
                err = {
                    "seed": seed,
                    "defense": name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                json.dump(err, open("results/ogbn_products_run_error.json", "w"), indent=2)
                print(err, flush=True)
                raise
            row["wall_seconds"] = round(time.time() - tw, 2)
            row["n_shadows_run"] = 2
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            print(
                seed,
                name,
                "acc",
                row["test_accuracy"],
                "conf",
                row["conf_attack_auc"],
                "lira",
                row["lira_attack_auc"],
                "wall",
                row["wall_seconds"],
                flush=True,
            )

    df = pd.DataFrame(rows)
    summary = {
        "dataset": "ogbn-products",
        "n_shadows": 2,
        "seeds": SEEDS,
        "n_nodes": probe.get("n_nodes"),
        "means": df.groupby("defense")[
            ["test_accuracy", "conf_attack_auc", "lira_attack_auc", "wall_seconds", "train_seconds"]
        ]
        .mean()
        .round(4)
        .to_dict(),
        "note": "Second Volume graph under official OGB split; none+SAMI.",
    }
    json.dump(summary, open(OUT_JSON, "w"), indent=2)
    probe["status"] = "audit_complete"
    probe["summary"] = summary
    json.dump(probe, open(PROBE, "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
