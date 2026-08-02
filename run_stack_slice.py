#!/usr/bin/env python3
"""Full clean-slice stack: ensemble constructor + deterministic confidence smoothing."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from defenses.harp import LOCKED_HARP_RELEASE, compute_harp_scales
from experiment import _load_target_data, _split_kwargs, _train_and_predict_gnn
from run_bulletproof import _cfg, _dcs

SEEDS5 = [42, 123, 456, 789, 1024]


def main():
    cfg = _cfg(4)
    device = torch.device("cpu")
    split_kw = _split_kwargs(cfg)
    ep = int(cfg.get("training", {}).get("epochs", 50))
    lr = float(cfg.get("training", {}).get("lr", 0.01))
    wd = float(cfg.get("training", {}).get("weight_decay", 5e-4))
    rows = []
    for seed in SEEDS5:
        params = {**LOCKED_HARP_RELEASE, "seed_mode": "ensemble"}
        data, nc, nf = _load_target_data("Cora", cfg["data_dir"], seed, True, split_kw)
        p, pr, risk, _, _, _ = _train_and_predict_gnn(
            "GraphSAGE", "harp_ensemble", params, data, nf, nc, device,
            ep, lr, wd, {}, None, False, 1024, [15, 10], cfg, release_seed=seed,
        )
        scales, prot, _, _ = compute_harp_scales(
            data.cpu(), risk=risk, risk_frac=params["risk_frac"], k_hops=1,
            strong_noise_scale=params["strong_noise_scale"], weak_noise_scale=0.0,
            target_protect_frac=params["target_protect_frac"], arch="sage", arch_aware=True,
        )
        unprot = ~np.asarray(prot, dtype=bool)
        yn = data.y.numpy(); trm = data.train_mask.numpy(); tem = data.test_mask.numpy()

        def slice_auc(pp):
            c = pp[np.arange(len(yn)), yn]
            mm, nn2 = trm & unprot, tem & unprot
            s = np.concatenate([c[mm], c[nn2]])
            y2 = np.concatenate([np.ones(int(mm.sum())), np.zeros(int(nn2.sum()))])
            return float(roc_auc_score(y2, s))

        base = slice_auc(p)
        p2, hot = _dcs(p, unprot, theta=0.90, temp=3.0)
        acc0 = float((p.argmax(1)[tem] == yn[tem]).mean())
        acc2 = float((p2.argmax(1)[tem] == yn[tem]).mean())
        rows.append({
            "seed": seed, "slice_ens": base, "slice_ens_dcs": slice_auc(p2),
            "Acc": acc0, "Acc_dcs": acc2,
            "argmax_ok": bool((p2.argmax(1) == p.argmax(1)).all()),
        })
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv("results/harp_stack_slice.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.mean(numeric_only=True).round(4), flush=True)
    print("STACK DONE", flush=True)


if __name__ == "__main__":
    main()
