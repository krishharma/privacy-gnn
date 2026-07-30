"""Run ogbn-arxiv Volume grid + systems metrics (train time, RSS, QPS)."""
from __future__ import annotations

import json
import os
import resource
import time

import pandas as pd
import torch

from config import ensure_dirs, load_config
from experiment import run_one
from graph_minibatch import measure_api_qps
from models import SAGE
from ogb_loader import load_large_benchmark
from training import train_gnn  # noqa: F401


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 10**9:
        return rss / (1024 * 1024)
    return rss / 1024.0


def main():
    os.environ["PRIVACYGNN_CONFIG"] = "experiment_config_ogbn.yaml"
    cfg = load_config()
    ensure_dirs(cfg)
    smoke_path = os.path.join(cfg["results_dir"], "ogbn_smoke_timing.json")
    if os.path.isfile(smoke_path):
        with open(smoke_path) as f:
            smoke = json.load(f)
        policy = smoke.get("policy", {})
        n_shadows = int(policy.get("n_shadows", cfg.get("lira", {}).get("n_shadows", 2)))
        # Prefer config seeds; cap by policy when present
        seeds = list(cfg.get("seeds", [42, 123, 456]))[: int(policy.get("n_seeds", 3))]
        cfg["lira"] = dict(cfg.get("lira", {}), n_shadows=min(n_shadows, int(cfg.get("lira", {}).get("n_shadows", 2))))
    else:
        seeds = [42, 123, 456]
        cfg["lira"] = dict(cfg.get("lira", {}), n_shadows=2)

    defenses = cfg.get("defenses", [("none", {})])
    rows = []
    device = torch.device(cfg.get("device", "cpu"))
    for seed in seeds:
        for d in defenses:
            if isinstance(d, dict):
                dname, dparams = d["name"], d.get("params", {})
            else:
                dname, dparams = d[0], d[1]
            t0 = time.time()
            row = run_one(
                "ogbn-arxiv",
                "GraphSAGE",
                dname,
                dparams,
                int(seed),
                config=cfg,
                device=device,
            )
            row["wall_seconds"] = round(time.time() - t0, 2)
            row["peak_rss_mb"] = round(peak_rss_mb(), 1)
            rows.append(row)
            print(
                f"done seed={seed} def={dname} acc={row['test_accuracy']} "
                f"conf={row['conf_attack_auc']} lira={row['lira_attack_auc']} "
                f"wall={row['wall_seconds']}s"
            )
            out_csv = os.path.join(cfg["results_dir"], "ogbn_volume_results.csv")
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Systems QPS on a fresh none model
    data, nc, nf = load_large_benchmark("ogbn-arxiv", os.path.join(cfg["data_dir"], "ogb"))
    model = SAGE(nf, 64, nc)
    from graph_minibatch import train_gnn_minibatch

    train_gnn_minibatch(
        model,
        data,
        device,
        data.edge_index,
        epochs=2,
        batch_size=1024,
        num_neighbors=[15, 10],
    )
    qps = measure_api_qps(model, data, device, [15, 10], 1024)
    systems = {
        "dataset": "ogbn-arxiv",
        "n_nodes": int(data.num_nodes),
        "n_edges": int(data.edge_index.size(1)),
        "api_qps_approx": round(qps, 1),
        "device": str(device),
        "n_rows": len(rows),
    }
    with open(os.path.join(cfg["results_dir"], "ogbn_systems.json"), "w") as f:
        json.dump(systems, f, indent=2)
    # Update scaling limitations
    with open(os.path.join(cfg["results_dir"], "scaling_limitations.json"), "w") as f:
        json.dump(
            {
                "ogbn_arxiv": "Enabled via NeighborLoader; see ogbn_volume_results.csv / ogbn_systems.json.",
                "actor_ok": True,
                "stretch_reddit_products": "Gated: only if Volume grid finishes ≥10 days before deadline.",
                "largest_graph": "ogbn-arxiv (~169k nodes).",
            },
            f,
            indent=2,
        )
    print(json.dumps(systems, indent=2))
    print(f"Wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
