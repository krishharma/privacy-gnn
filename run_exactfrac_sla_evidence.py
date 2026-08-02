#!/usr/bin/env python3
"""
ExactFrac SLA evidence: sticky noise ≠ ExactFrac.

Measures what production serving cares about:
  (A) cross-client bit-equality under identical node queries
  (B) pre-session audit replay (response logged before session vs after)
  (C) threshold-router flicker on the clean majority
  (D) shared content-cache hit rate across clients (Zipf)

Sticky LBP (B=1 per client) restores per-client hits but fails (A)/(B)/(D).
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from defenses.harp import LOCKED_HARP_RELEASE, compute_harp_scales
from defenses.lbp import lbp_perturb
from defenses.sami import risk_scaled_posterior_noise
from experiment import _load_target_data, _split_kwargs, _train_and_predict_gnn
from run_bulletproof import _cfg

OUT = "results"
SEEDS = [42, 123, 456]


class LRU:
    def __init__(self, cap: int):
        self.cap = int(cap)
        self.d: OrderedDict = OrderedDict()

    def get(self, k):
        if k not in self.d:
            return None
        self.d.move_to_end(k)
        return self.d[k]

    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)


def _key(p_row: np.ndarray) -> bytes:
    return np.ascontiguousarray(p_row).tobytes()


def main():
    cfg = _cfg(4)
    path = os.path.join(OUT, "harp_exactfrac_sla_evidence.csv")
    rows = []
    device = torch.device("cpu")
    split_kw = _split_kwargs(cfg)
    ep = int(cfg.get("training", {}).get("epochs", 50))
    lr = float(cfg.get("training", {}).get("lr", 0.01))
    wd = float(cfg.get("training", {}).get("weight_decay", 5e-4))
    theta = 0.5
    n_clients = 8
    n_req = 20000
    cache_cap = 1024

    for seed in SEEDS:
        print(f"SLAEV seed={seed}", flush=True)
        data, nc, nf = _load_target_data("Cora", cfg["data_dir"], seed, True, split_kw)
        n = int(data.num_nodes)
        p_base, _, risk, _, _, _ = _train_and_predict_gnn(
            "GraphSAGE", "harp_release_only",
            {**LOCKED_HARP_RELEASE, "strong_noise_scale": 0.0},
            data, nf, nc, device, ep, lr, wd, {}, None, False, 1024, [15, 10], cfg,
            release_seed=seed,
        )
        scales, prot, _, _ = compute_harp_scales(
            data.cpu(), risk=risk, risk_frac=0.30, k_hops=1,
            strong_noise_scale=0.30, weak_noise_scale=0.0,
            target_protect_frac=0.40, arch="sage", arch_aware=True,
        )
        scales = np.asarray(scales)
        clean = ~np.asarray(prot, dtype=bool)

        def harp(qseed):
            return risk_scaled_posterior_noise(p_base, scales, 1.0, seed=qseed)

        def lbp_fresh(qseed):
            return lbp_perturb(p_base, scale=0.12, n_bins=None, seed=qseed)

        # --- (A) cross-client ExactFrac: fraction of nodes identical across 2 clients ---
        for policy, rel in [("none", lambda s: p_base), ("lbp_fresh", lbp_fresh),
                            ("lbp_sticky", lbp_fresh), ("harp", harp)]:
            # sticky: each client freezes first draw; cross-client compares frozen maps
            if policy == "lbp_sticky":
                p_a = lbp_fresh(seed * 100 + 1)
                p_b = lbp_fresh(seed * 100 + 2)
            elif policy == "none":
                p_a = p_b = p_base
            else:
                p_a = rel(seed * 100 + 1)
                p_b = rel(seed * 100 + 2)
            cross = float((p_a == p_b).all(axis=1).mean())
            # --- (B) pre-session audit: "logged" answer before session vs sticky session ---
            p_pre = rel(seed * 1000) if policy != "none" else p_base
            if policy == "lbp_sticky":
                p_sess = lbp_fresh(seed * 100 + 1)  # client A's sticky map
            else:
                p_sess = p_a
            audit_replay = float((p_pre == p_sess).all(axis=1).mean())
            # --- (C) clean-slice threshold flicker across two independent releases ---
            if policy == "none":
                th_clean = 0.0
            elif policy == "lbp_sticky":
                # sticky: same client, no flicker; but clean-slice of *other* client differs
                th_clean = float(
                    ((p_a.max(1) > theta) != (p_b.max(1) > theta))[clean].mean()
                )
            else:
                th_clean = float(
                    ((p_a.max(1) > theta) != (p_b.max(1) > theta))[clean].mean()
                )
            # --- (D) shared cache across clients ---
            rng = np.random.RandomState(seed + 7)
            # Zipf node ids
            weights = 1.0 / np.arange(1, n + 1) ** 1.1
            weights /= weights.sum()
            cache = LRU(cache_cap)
            hits = 0
            sticky_maps = {}
            for t in range(n_req):
                cid = int(rng.randint(0, n_clients))
                v = int(rng.choice(n, p=weights))
                if policy == "none":
                    row = p_base[v]
                elif policy == "lbp_fresh":
                    row = lbp_fresh(seed * 10_000 + t)[v]
                elif policy == "lbp_sticky":
                    if cid not in sticky_maps:
                        sticky_maps[cid] = lbp_fresh(seed * 100 + cid)
                    row = sticky_maps[cid][v]
                else:  # harp: clean nodes shared; protected per-draw (use client sticky on prot)
                    if cid not in sticky_maps:
                        sticky_maps[cid] = harp(seed * 100 + cid)
                    row = sticky_maps[cid][v]
                k = (v, _key(row))
                if cache.get(k) is not None:
                    hits += 1
                else:
                    cache.put(k, 1)
            hit_rate = hits / n_req
            rows.append({
                "seed": seed, "policy": policy,
                "cross_client_exactfrac": cross,
                "presession_audit_replay": audit_replay,
                "clean_threshold_flicker": th_clean,
                "shared_cache_hit": hit_rate,
                "n_clients": n_clients, "n_req": n_req, "cache_cap": cache_cap,
            })
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    df = pd.DataFrame(rows)
    means = df.groupby("policy")[
        ["cross_client_exactfrac", "presession_audit_replay",
         "clean_threshold_flicker", "shared_cache_hit"]
    ].agg(["mean", "std"])
    print(means.round(4), flush=True)
    means.to_csv(os.path.join(OUT, "harp_exactfrac_sla_evidence_means.csv"))
    print("SLAEV DONE", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PRIVACYGNN_CONFIG", "experiment_config_confirmatory.yaml")
    main()
