#!/usr/bin/env python3
"""Extend session B-sweep + adaptive adversary to 5 seeds (append 789, 1024)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from attacks import average_posterior_queries, calibration_error
from config import ensure_dirs, load_config
from defenses.harp import LOCKED_HARP, compute_harp_scales
from defenses.sami import risk_scaled_posterior_noise
from experiment import (
    _load_target_data,
    _make_shadow_data,
    _split_kwargs,
    _train_and_predict_gnn,
)
from lira_attack import lira_auc_on_subset, lira_gaussian_auc, lira_gaussian_scores
from run_harp_adaptive_adversary import _conf_auc, _neighbors

EXTRA = [789, 1024]


def main():
    cfg = dict(load_config())
    ensure_dirs(cfg)
    cfg["lira"] = {"n_shadows": 4}
    device = torch.device(cfg.get("device", "cpu"))
    split_kw = _split_kwargs(cfg)
    ep = int(cfg.get("training", {}).get("epochs", 50))
    lr = float(cfg.get("training", {}).get("lr", 0.01))
    wd = float(cfg.get("training", {}).get("weight_decay", 5e-4))
    K = 20

    out_b = "results/harp_session_b_sweep_cora.csv"
    rows_b = pd.read_csv(out_b).to_dict("records") if os.path.isfile(out_b) else []
    done_b = {(int(r["seed"]), r["policy"], int(r["B"]), int(r["K"])) for r in rows_b}
    for seed in EXTRA:
        print(f"B-SWEEP seed={seed}", flush=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        data, nc, nf = _load_target_data("Cora", cfg["data_dir"], seed, True, split_kw)
        p_base, pr, risk, train_s, _, _ = _train_and_predict_gnn(
            "GraphSAGE",
            "harp",
            {**LOCKED_HARP, "strong_noise_scale": 0.0, "weak_noise_scale": 0.0},
            data,
            nf,
            nc,
            device,
            ep,
            lr,
            wd,
            {},
            None,
            False,
            1024,
            [15, 10],
            cfg,
            release_seed=seed,
            multi_query_k=1,
        )
        scales, prot, _, _ = compute_harp_scales(
            data.cpu(),
            risk=risk,
            risk_frac=LOCKED_HARP["risk_frac"],
            k_hops=LOCKED_HARP["k_hops"],
            strong_noise_scale=LOCKED_HARP["strong_noise_scale"],
            weak_noise_scale=0.0,
            target_protect_frac=LOCKED_HARP["target_protect_frac"],
            arch="sage",
            arch_aware=True,
        )
        scales = np.asarray(scales, dtype=float)
        yn = data.y.numpy()
        trm = data.train_mask.numpy()
        tem = data.test_mask.numpy()
        risk_bin = (scales > 1e-12).astype(float)
        strong = float(LOCKED_HARP["strong_noise_scale"])
        shadow_probs, shadow_tr, shadow_te = [], [], []
        for k in range(4):
            sdata, _, _ = _make_shadow_data("Cora", cfg["data_dir"], seed + 100 + k, split_kw)
            sp, _, _, _, _, _ = _train_and_predict_gnn(
                "GraphSAGE",
                "harp",
                dict(LOCKED_HARP),
                sdata,
                nf,
                nc,
                device,
                ep,
                lr,
                wd,
                {},
                None,
                False,
                1024,
                [15, 10],
                cfg,
                release_seed=seed + 100 + k,
                multi_query_k=1,
            )
            shadow_probs.append(sp)
            shadow_tr.append(sdata.train_mask.numpy())
            shadow_te.append(sdata.test_mask.numpy())
        for B in [1, 5, 20]:
            for policy in ["uncapped_avg", "budget_reuse"]:
                if (seed, policy, B, K) in done_b:
                    continue
                if policy == "uncapped_avg":
                    if B != 5:
                        continue
                    p = average_posterior_queries(p_base, risk_bin, strong, K, seed0=seed)
                else:
                    bb = min(K, B)
                    draws = [
                        risk_scaled_posterior_noise(p_base, risk_bin, strong, seed=seed + t)
                        for t in range(bb)
                    ]
                    while len(draws) < K:
                        draws.append(draws[-1])
                    p = np.mean(draws[:K], axis=0)
                pr2 = p.argmax(1)
                acc = float((pr2[tem] == yn[tem]).mean())
                lira, _, _, _ = lira_gaussian_auc(
                    p, yn, trm, tem, shadow_probs, shadow_tr, shadow_te
                )
                ece = calibration_error(p[tem], yn[tem])
                row = {
                    "seed": seed,
                    "policy": policy,
                    "B": B,
                    "K": K,
                    "acc": acc,
                    "lira": float(lira),
                    "ece": float(ece),
                    "train_seconds": float(train_s),
                }
                rows_b.append(row)
                done_b.add((seed, policy, B, K))
                print(f"  {policy} B={B}: Acc={acc:.3f} LiRA={lira:.3f}", flush=True)
        pd.DataFrame(rows_b).to_csv(out_b, index=False)
    print(pd.DataFrame(rows_b).groupby(["policy", "B"])[["acc", "lira"]].mean().round(4), flush=True)

    out_a = "results/harp_adaptive_adversary.csv"
    rows_a = pd.read_csv(out_a).to_dict("records") if os.path.isfile(out_a) else []
    done_a = {int(r["seed"]) for r in rows_a}
    for seed in EXTRA:
        if seed in done_a:
            continue
        print(f"ADAPTIVE seed={seed}", flush=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        data, nc, nf = _load_target_data("Cora", cfg["data_dir"], seed, True, split_kw)
        p, pr, risk, train_s, _, _ = _train_and_predict_gnn(
            "GraphSAGE",
            "harp",
            dict(LOCKED_HARP),
            data,
            nf,
            nc,
            device,
            ep,
            lr,
            wd,
            {},
            None,
            False,
            1024,
            [15, 10],
            cfg,
            release_seed=seed,
            multi_query_k=1,
        )
        scales, prot, seeds_mask, hstats = compute_harp_scales(
            data.cpu(),
            risk=risk,
            risk_frac=LOCKED_HARP["risk_frac"],
            k_hops=LOCKED_HARP["k_hops"],
            strong_noise_scale=LOCKED_HARP["strong_noise_scale"],
            weak_noise_scale=LOCKED_HARP.get("weak_noise_scale", 0.0),
            target_protect_frac=LOCKED_HARP["target_protect_frac"],
            arch="sage",
            arch_aware=True,
        )
        prot = np.asarray(prot, dtype=bool)
        unprot = ~prot
        yn = data.y.numpy()
        trm = data.train_mask.numpy()
        tem = data.test_mask.numpy()
        n = len(yn)
        adj = _neighbors(data.edge_index.numpy(), n)
        boundary = np.zeros(n, dtype=bool)
        for v in np.where(unprot)[0]:
            if any(prot[u] for u in adj[v]):
                boundary[v] = True
        risk_np = np.asarray(risk if not hasattr(risk, "numpy") else risk.numpy(), dtype=float)
        thr = float(np.quantile(risk_np[trm], 0.9))
        top_lte = risk_np >= thr
        query_sets = {
            "population": np.ones(n, dtype=bool),
            "unprot_clean": unprot,
            "prot_boundary_clean": boundary,
            "top_decile_lte": top_lte,
        }
        conf = p[np.arange(n), yn]
        shadow_probs, shadow_tr, shadow_te = [], [], []
        for k in range(4):
            shadow_seed = int(seed + 999 + k * 10007)
            np.random.seed(shadow_seed)
            torch.manual_seed(shadow_seed)
            sdata, _, _ = _make_shadow_data("Cora", cfg["data_dir"], shadow_seed, split_kw)
            p_sh, _, _, _, _, _ = _train_and_predict_gnn(
                "GraphSAGE",
                "harp",
                dict(LOCKED_HARP),
                sdata,
                nf,
                nc,
                device,
                ep,
                lr,
                wd,
                {},
                None,
                False,
                1024,
                [15, 10],
                cfg,
                release_seed=shadow_seed,
                multi_query_k=1,
            )
            shadow_probs.append(p_sh)
            shadow_tr.append(sdata.train_mask.numpy())
            shadow_te.append(sdata.test_mask.numpy())
        scores, y_mem, _ = lira_gaussian_scores(
            p, yn, trm, tem, shadow_probs, shadow_tr, shadow_te
        )
        lira_pop, _, tpr001, tpr01 = lira_gaussian_auc(
            p, yn, trm, tem, shadow_probs, shadow_tr, shadow_te
        )
        acc = float((pr[tem] == yn[tem]).mean())
        for name, qmask in query_sets.items():
            c_auc, n_m, n_n = _conf_auc(conf, trm, tem, qmask)
            if name == "population":
                l_auc = float(lira_pop)
            else:
                l_auc, _, _ = lira_auc_on_subset(
                    scores, y_mem, np.where(trm & qmask)[0], np.where(tem & qmask)[0]
                )
            rows_a.append(
                {
                    "seed": seed,
                    "query_set": name,
                    "acc": acc,
                    "frac_prot": float(hstats["frac_protected"]),
                    "conf_auc": c_auc,
                    "lira_auc": float(l_auc),
                    "lira_tpr_at_0.001_fpr": float(tpr001) if name == "population" else float("nan"),
                    "lira_tpr_at_0.01_fpr": float(tpr01) if name == "population" else float("nan"),
                    "n_member_queries": n_m,
                    "n_nonmember_queries": n_n,
                    "train_seconds": float(train_s),
                }
            )
            print(rows_a[-1], flush=True)
        pd.DataFrame(rows_a).to_csv(out_a, index=False)
    print(pd.DataFrame(rows_a).groupby("query_set")[["conf_auc", "lira_auc"]].mean().round(4), flush=True)
    print("5SEED EXTEND DONE", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PRIVACYGNN_CONFIG", "experiment_config_confirmatory.yaml")
    main()
