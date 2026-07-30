"""
LTE-/HARP-aware adaptive query adversary.

An adversary who knows selective release is used concentrates queries on:
  (1) unprotected (clean) nodes
  (2) clean nodes adjacent to the protected set (hop boundary)
  (3) top-decile LTE seeds among train members (worst-case pocket)

Reports confidence AUROC and LiRA AUROC on each query set vs population.
Converts the Section III-B caveat into a measured result.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from config import ensure_dirs, load_config
from defenses.harp import LOCKED_HARP, compute_harp_scales
from experiment import _load_target_data, _make_shadow_data, _split_kwargs, _train_and_predict_gnn
from lira_attack import lira_auc_on_subset, lira_gaussian_auc, lira_gaussian_scores

SEEDS = [42, 123, 456]
OUT = "results/harp_adaptive_adversary.csv"
N_SHADOWS = 4


def _neighbors(edge_index: np.ndarray, n: int) -> list[set[int]]:
    adj = [set() for _ in range(n)]
    src, dst = edge_index[0], edge_index[1]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _conf_auc(conf, trm, tem, qmask):
    m = trm & qmask
    n = tem & qmask
    if m.sum() < 5 or n.sum() < 5:
        return float("nan"), int(m.sum()), int(n.sum())
    s = np.concatenate([conf[m], conf[n]])
    y = np.concatenate([np.ones(int(m.sum())), np.zeros(int(n.sum()))])
    return float(roc_auc_score(y, s)), int(m.sum()), int(n.sum())


def main():
    os.environ.setdefault("PRIVACYGNN_CONFIG", "experiment_config_confirmatory.yaml")
    cfg = dict(load_config())
    ensure_dirs(cfg)
    cfg["lira"] = {"n_shadows": N_SHADOWS}
    device = torch.device(cfg.get("device", "cpu"))
    split_kw = _split_kwargs(cfg)
    ep = int(cfg.get("training", {}).get("epochs", 50))
    lr = float(cfg.get("training", {}).get("lr", 0.01))
    wd = float(cfg.get("training", {}).get("weight_decay", 5e-4))
    rows = []

    for seed in SEEDS:
        print(f"ADAPTIVE seed={seed}", flush=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        data, nc, nf = _load_target_data("Cora", cfg["data_dir"], seed, True, split_kw)
        p, pr, risk, train_s, _, _ = _train_and_predict_gnn(
            "GraphSAGE", "harp", dict(LOCKED_HARP), data, nf, nc, device,
            ep, lr, wd, {}, None, False, 1024, [15, 10], cfg,
            release_seed=seed, multi_query_k=1,
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
        for k in range(N_SHADOWS):
            shadow_seed = int(seed + 999 + k * 10007)
            np.random.seed(shadow_seed)
            torch.manual_seed(shadow_seed)
            sdata, _, _ = _make_shadow_data("Cora", cfg["data_dir"], shadow_seed, split_kw)
            p_sh, _, _, _, _, _ = _train_and_predict_gnn(
                "GraphSAGE", "harp", dict(LOCKED_HARP), sdata, nf, nc, device,
                ep, lr, wd, {}, None, False, 1024, [15, 10], cfg,
                release_seed=shadow_seed, multi_query_k=1,
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
            rows.append({
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
            })
            print(rows[-1], flush=True)

        pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    print("\nMeans by query_set:")
    print(df.groupby("query_set")[["conf_auc", "lira_auc"]].mean())
    return df


if __name__ == "__main__":
    main()
