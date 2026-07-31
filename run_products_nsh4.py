#!/usr/bin/env python3
"""Products BFS-subgraph LiRA at n_shadows=4, 5 seeds (closes scale-audit gap)."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from config import load_config
from defenses.harp import LOCKED_HARP_RELEASE
from experiment import _train_and_predict_gnn
from lira_attack import _logit_confidence

OUT = "results"
SEEDS5 = [42, 123, 456, 789, 1024]


def main(n_sub=15000, n_shadows=4):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/harp_products_sub{n_sub}_nsh{n_shadows}.csv"
    sub_path = f"{OUT}/products_sub_{n_sub}.pt"
    cfg = load_config("experiment_config_ogbn.yaml")
    cfg["lira"] = {"n_shadows": n_shadows}
    device = torch.device("cpu")
    assert os.path.isfile(sub_path), f"missing {sub_path}"
    base = torch.load(sub_path, weights_only=False)["data"]
    rows = []
    if os.path.isfile(path):
        rows = pd.read_csv(path).to_dict("records")
    done = {(r["tag"], int(r["seed"])) for r in rows}
    harp = {**LOCKED_HARP_RELEASE, "use_gate": False, "warmup_epochs": 3}
    jobs = [
        ("none", "none", {}),
        ("lbp", "lbp", {"scale": 0.3}),
        ("harp", "harp_release_only", harp),
    ]
    for tag, dn, dp in jobs:
        for seed in SEEDS5:
            if (tag, seed) in done:
                print("skip", tag, seed, flush=True)
                continue
            print(f"PRODUCTS n={n_sub} nsh={n_shadows} {tag} seed={seed}", flush=True)
            t0 = time.time()
            data = base.clone()
            m = data.num_nodes
            rng = np.random.RandomState(seed)
            perm = rng.permutation(m)
            n_tr, n_va = int(0.4 * m), int(0.2 * m)
            tr = torch.zeros(m, dtype=torch.bool)
            va = torch.zeros(m, dtype=torch.bool)
            te = torch.zeros(m, dtype=torch.bool)
            tr[perm[:n_tr]] = True
            va[perm[n_tr : n_tr + n_va]] = True
            te[perm[n_tr + n_va :]] = True
            data.train_mask, data.val_mask, data.test_mask = tr, va, te
            nf = int(data.x.size(1))
            nc = int(data.y.max().item()) + 1
            tk = {
                "epochs": 30, "lr": 0.01, "weight_decay": 5e-4, "device": "cpu",
                "early_stop_patience": None, "label_smoothing": 0.0,
                "dropedge_rate": 0.0, "edge_sparsify_rate": 0.0,
            }
            p, pr, _, _, _, rel = _train_and_predict_gnn(
                "GraphSAGE", dn, dp, data, nf, nc, device,
                tk["epochs"], tk["lr"], tk["weight_decay"], tk, None, False, 1024, [25, 25],
                cfg, release_seed=seed,
            )
            yn = data.y.view(-1).cpu().numpy()
            conf_t = _logit_confidence(p, yn)
            in_mu = np.zeros(m); out_mu = np.zeros(m); in_n = np.zeros(m); out_n = np.zeros(m)
            for k in range(n_shadows):
                sdata = data.clone()
                rng2 = np.random.RandomState(seed + 1000 + k)
                perm2 = rng2.permutation(m)
                tr2 = torch.zeros(m, dtype=torch.bool)
                tr2[perm2[:n_tr]] = True
                sdata.train_mask = tr2
                sp, _, _, _, _, _ = _train_and_predict_gnn(
                    "GraphSAGE", dn, dp, sdata, nf, nc, device,
                    tk["epochs"], tk["lr"], tk["weight_decay"], tk, None, False, 1024, [25, 25],
                    cfg, release_seed=seed + k,
                )
                conf = _logit_confidence(sp, yn)
                sm = tr2.cpu().numpy()
                in_mu += conf * sm; in_n += sm
                out_mu += conf * (~sm); out_n += (~sm)
            in_mu /= np.maximum(in_n, 1); out_mu /= np.maximum(out_n, 1)
            score = -np.abs(conf_t - in_mu) + np.abs(conf_t - out_mu)
            trn = tr.cpu().numpy(); ten = te.cpu().numpy()
            mask = trn | ten
            la = float(roc_auc_score(trn[mask].astype(int), score[mask]))
            acc = float((pr[ten] == yn[ten]).mean())
            rows.append({
                "tag": tag, "seed": seed, "n_sub": n_sub, "n_shadows": n_shadows,
                "Acc": acc, "LiRA": la,
                "Mass": rel.get("noise_mass"), "Frac": rel.get("frac_protected"),
                "wall": round(time.time() - t0, 2),
            })
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  Acc={acc:.3f} LiRA={la:.3f} wall={rows[-1]['wall']}", flush=True)
    df = pd.DataFrame(rows)
    means = df.groupby("tag")[["Acc", "LiRA"]].agg(["mean", "std", "count"])
    means.to_csv(f"{OUT}/harp_products_sub{n_sub}_nsh{n_shadows}_means.csv")
    print(means, flush=True)


if __name__ == "__main__":
    main()
