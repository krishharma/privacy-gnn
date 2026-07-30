"""Reduced shadow-count stability sweep for paper Table audit(b).
Cora: n_shadows in {4,16,64}; Chameleon: {4,16}. Defenses: none, lbp, harp.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import torch

from config import ensure_dirs, load_config
from defenses.harp import LOCKED_HARP
from experiment import run_one

SEEDS = [42, 123, 456]
OUT = "results/harp_shadow_sweep.csv"


def main():
    os.environ.setdefault("PRIVACYGNN_CONFIG", "experiment_config_confirmatory.yaml")
    cfg = dict(load_config())
    ensure_dirs(cfg)
    cfg["attacks"] = ["confidence", "lira"]
    device = torch.device(cfg.get("device", "cpu"))
    rows = []
    done = set()
    if os.path.isfile(OUT):
        old = pd.read_csv(OUT)
        done = {(r.dataset, r.tag, int(r.n_shadows_run), int(r.seed)) for _, r in old.iterrows()}
        rows = old.to_dict("records")
    for ds, n_list in [("Cora", [4, 16, 64]), ("Chameleon", [4, 16])]:
        for n_sh in n_list:
            for tag, dn, dp in [
                ("none", "none", {}),
                ("lbp", "lbp", {"scale": 0.3}),
                ("harp", "harp", dict(LOCKED_HARP)),
            ]:
                for seed in SEEDS:
                    key = (ds, tag, n_sh, seed)
                    if key in done:
                        print("skip", key, flush=True)
                        continue
                    print(f"SHADOW {ds} {tag} n={n_sh} seed={seed}", flush=True)
                    local = dict(cfg)
                    local["lira"] = {"n_shadows": int(n_sh)}
                    t0 = time.time()
                    r = run_one(ds, "GraphSAGE", dn, dp, seed, device=device, config=local)
                    r = dict(r)
                    r["tag"] = tag
                    r["n_shadows_run"] = int(n_sh)
                    r["wall_seconds"] = round(time.time() - t0, 2)
                    rows.append(r)
                    pd.DataFrame(rows).to_csv(OUT, index=False)
                    print(
                        f"  acc={r['test_accuracy']:.4f} lira={r['lira_attack_auc']:.4f} "
                        f"tpr01={r.get('lira_tpr_at_0.01_fpr')} wall={r['wall_seconds']}",
                        flush=True,
                    )
    df = pd.read_csv(OUT)
    means = (
        df.groupby(["dataset", "tag", "n_shadows_run"])[
            ["test_accuracy", "lira_attack_auc", "lira_tpr_at_0.01_fpr", "lira_tpr_at_0.001_fpr"]
        ]
        .mean()
        .round(4)
    )
    means.to_csv("results/harp_shadow_sweep_means.csv")
    print(means, flush=True)


if __name__ == "__main__":
    main()
