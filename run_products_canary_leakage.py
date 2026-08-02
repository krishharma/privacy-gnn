#!/usr/bin/env python3
"""Canary-stressed ogbn-products BFS-15k LiRA probe (above-chance privacy signal)."""
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
SEEDS = [42, 123, 456]


def plant(data, seed, n_can=300, train_frac=0.10):
    d = data.clone()
    m = d.num_nodes
    rng = np.random.RandomState(seed)
    idx = rng.choice(m, n_can, replace=False)
    extra = torch.zeros(m, 8)
    nc = int(d.y.max().item()) + 1
    for j, i in enumerate(idx):
        extra[i, j % 8] = 50.0
        d.y[i] = j % nc
    d.x = torch.cat([d.x, extra], dim=1)
    rest = [i for i in rng.permutation(m).tolist() if i not in set(idx.tolist())]
    n_tr = int(train_frac * m)
    n_va = int(0.1 * m)
    tr = torch.zeros(m, dtype=torch.bool)
    va = torch.zeros(m, dtype=torch.bool)
    te = torch.zeros(m, dtype=torch.bool)
    rng.shuffle(idx)
    in_can = idx[: n_can // 2]
    out_can = idx[n_can // 2 :]
    tr[in_can] = True
    need = n_tr - len(in_can)
    tr[rest[:need]] = True
    rem = [i for i in rest[need:] if not tr[i]]
    te[out_can] = True
    rem2 = [i for i in rem if i not in set(out_can.tolist())]
    va[rem2[:n_va]] = True
    te[rem2[n_va:]] = True
    d.train_mask, d.val_mask, d.test_mask = tr, va, te
    return d, in_can, out_can


def main():
    path = os.path.join(OUT, "harp_products_canary_leakage.csv")
    base = torch.load("results/products_sub_15000.pt", weights_only=False)["data"]
    cfg = load_config("experiment_config_ogbn.yaml")
    n_sh = 4
    epochs = 120
    train_frac = 0.10
    device = torch.device("cpu")
    rows = pd.read_csv(path).to_dict("records") if os.path.isfile(path) else []
    done = {(r["tag"], int(r["seed"])) for r in rows}
    harp = {**LOCKED_HARP_RELEASE, "use_gate": False, "warmup_epochs": 3}
    jobs = [
        ("none", "none", {}),
        ("lbp", "lbp", {"scale": 0.3}),
        ("harp", "harp_release_only", harp),
    ]
    tk = {
        "epochs": epochs,
        "lr": 0.01,
        "weight_decay": 0.0,
        "device": "cpu",
        "early_stop_patience": None,
        "label_smoothing": 0.0,
        "dropedge_rate": 0.0,
        "edge_sparsify_rate": 0.0,
    }
    for tag, dn, dp in jobs:
        for seed in SEEDS:
            if (tag, seed) in done:
                continue
            print(f"CANARY {tag} seed={seed}", flush=True)
            t0 = time.time()
            data, in_can, out_can = plant(base, seed)
            m = data.num_nodes
            nf = int(data.x.size(1))
            nc = int(data.y.max().item()) + 1
            n_tr = int(train_frac * m)
            p, pr, _, _, _, _ = _train_and_predict_gnn(
                "GraphSAGE", dn, dp, data, nf, nc, device,
                epochs, 0.01, 0.0, tk, None, False, 1024, [25, 25],
                cfg, release_seed=seed,
            )
            yn = data.y.view(-1).cpu().numpy()
            conf_t = _logit_confidence(p, yn)
            in_mu = np.zeros(m)
            out_mu = np.zeros(m)
            in_n = np.zeros(m)
            out_n = np.zeros(m)
            for k in range(n_sh):
                sdata = data.clone()
                rng2 = np.random.RandomState(seed + 1000 + k)
                perm2 = rng2.permutation(m)
                tr2 = torch.zeros(m, dtype=torch.bool)
                tr2[perm2[:n_tr]] = True
                sdata.train_mask = tr2
                sp, _, _, _, _, _ = _train_and_predict_gnn(
                    "GraphSAGE", dn, dp, sdata, nf, nc, device,
                    epochs, 0.01, 0.0, tk, None, False, 1024, [25, 25],
                    cfg, release_seed=seed + k,
                )
                conf = _logit_confidence(sp, yn)
                sm = tr2.cpu().numpy()
                in_mu += conf * sm
                in_n += sm
                out_mu += conf * (~sm)
                out_n += (~sm)
            in_mu /= np.maximum(in_n, 1)
            out_mu /= np.maximum(out_n, 1)
            score = -np.abs(conf_t - in_mu) + np.abs(conf_t - out_mu)
            trn = data.train_mask.cpu().numpy()
            ten = data.test_mask.cpu().numpy()
            mask = trn | ten
            la = float(roc_auc_score(trn[mask].astype(int), score[mask]))
            y_c = np.concatenate([np.ones(len(in_can)), np.zeros(len(out_can))])
            s_c = np.concatenate([score[in_can], score[out_can]])
            can_lira = float(roc_auc_score(y_c, s_c))
            acc = float((pr[ten] == yn[ten]).mean())
            rows.append({
                "tag": tag, "seed": seed, "n_sub": 15000, "n_shadows": n_sh,
                "Acc": acc, "LiRA": la, "canary_LiRA": can_lira,
                "ExactFrac": {"none": 1.0, "lbp": 0.0, "harp": 0.60}[tag],
                "wall": round(time.time() - t0, 1),
            })
            done.add((tag, seed))
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  Acc={acc:.3f} LiRA={la:.3f} canLiRA={can_lira:.3f}", flush=True)
    print(pd.DataFrame(rows).groupby("tag")[["Acc", "LiRA", "canary_LiRA"]].agg(["mean", "std"]))


if __name__ == "__main__":
    os.environ.setdefault("PRIVACYGNN_CONFIG", "experiment_config_ogbn.yaml")
    main()
