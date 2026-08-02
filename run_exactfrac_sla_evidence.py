#!/usr/bin/env python3
"""
RawExactFrac vs ReplayFrac SLA evidence.

Shows that stateful response caching / sticky noise can raise ReplayFrac
without providing RawExactFrac (equality to the unmodified posterior).

Policies:
  none            — raw posteriors
  lbp_fresh       — independent Laplace each query
  lbp_sticky      — per-client first-draw freeze (B=1)
  lbp_global      — shared global first-draw cache keyed by node id
  lbp_seeded      — deterministic PRNG Laplace keyed by (node, epoch)
  harp            — selective release (Frac=0.40, weak=0)

Metrics:
  raw_exactfrac   — fraction of nodes with tilde p_v == hat p_v
  replay_frac     — fraction of nodes identical across two independent draws
  cross_client    — equality across two clients / independent maps
  audit_replay    — pre-session logged answer vs later serving map
  clean_flicker   — threshold flip rate on HARP's clean majority
  shared_cache_hit— Zipf traffic into shared content LRU
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
EPOCH_KEY = 7  # release-epoch salt for seeded noise


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


def _row_equal(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Per-node equality tolerant to float renormalization (~1e-7)."""
    return np.all(np.abs(a - b) <= atol, axis=1)


def _seeded_lbp(p_base: np.ndarray, scale: float = 0.12) -> np.ndarray:
    """Deterministic Laplace keyed by (node_id, EPOCH_KEY); RawExactFrac=0, ReplayFrac=1."""
    n, c = p_base.shape
    out = np.empty_like(p_base)
    for v in range(n):
        rng = np.random.RandomState((int(v) * 1_000_003 + EPOCH_KEY) % (2**31 - 1))
        # Match LBP binning: one Laplace draw per confidence rank bin.
        row = p_base[v].copy()
        order = np.argsort(-row)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(c)
        noise = rng.laplace(0.0, scale, size=c)
        noisy = np.maximum(row + noise[ranks], 0.0)
        s = noisy.sum()
        out[v] = noisy / s if s > 0 else np.ones(c) / c
    return out


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

        p_seeded = _seeded_lbp(p_base, scale=0.12)
        # Global first-draw cache: one shared map for all clients.
        p_global = lbp_fresh(seed * 777)

        policies = [
            ("none", lambda: p_base, lambda: p_base),
            ("lbp_fresh", lambda: lbp_fresh(seed * 100 + 1), lambda: lbp_fresh(seed * 100 + 2)),
            ("lbp_sticky", lambda: lbp_fresh(seed * 100 + 1), lambda: lbp_fresh(seed * 100 + 2)),
            ("lbp_global", lambda: p_global, lambda: p_global),
            ("lbp_seeded", lambda: p_seeded, lambda: p_seeded),
            ("harp", lambda: harp(seed * 100 + 1), lambda: harp(seed * 100 + 2)),
        ]

        for policy, draw_a, draw_b in policies:
            p_a = draw_a()
            p_b = draw_b()
            raw_ef = float(_row_equal(p_a, p_base).mean())
            # ReplayFrac: two independent draws (for sticky/global/seeded, draws are stateful/deterministic)
            if policy == "lbp_fresh":
                p_r1 = lbp_fresh(seed * 9001)
                p_r2 = lbp_fresh(seed * 9002)
                replay = float(_row_equal(p_r1, p_r2).mean())
            elif policy == "harp":
                p_r1 = harp(seed * 9001)
                p_r2 = harp(seed * 9002)
                replay = float(_row_equal(p_r1, p_r2).mean())
            else:
                # none / sticky maps / global / seeded: same map ⇒ replay 1 (or cross-client for sticky)
                if policy == "lbp_sticky":
                    # within one client sticky map: replay 1; we report within-client replay
                    replay = 1.0
                else:
                    replay = float(_row_equal(p_a, p_b).mean())

            cross = float(_row_equal(p_a, p_b).mean())

            # Pre-session audit: logged answer before "session" vs later serving
            if policy == "none":
                p_pre = p_base
                p_sess = p_base
            elif policy == "lbp_fresh":
                p_pre = lbp_fresh(seed * 1000)
                p_sess = lbp_fresh(seed * 100 + 1)
            elif policy == "lbp_sticky":
                p_pre = lbp_fresh(seed * 1000)  # audit before session
                p_sess = p_a  # client sticky map
            elif policy == "lbp_global":
                p_pre = p_global  # ledger / cache is the audit source of truth
                p_sess = p_global
            elif policy == "lbp_seeded":
                p_pre = p_seeded
                p_sess = p_seeded
            else:  # harp: clean majority matches across independent draws (no ledger)
                p_pre = harp(seed * 1000)
                p_sess = p_a
            audit_replay = float(_row_equal(p_pre, p_sess).mean())

            if policy in ("none", "lbp_global", "lbp_seeded"):
                th_clean = 0.0
            elif policy == "lbp_sticky":
                th_clean = float(
                    ((p_a.max(1) > theta) != (p_b.max(1) > theta))[clean].mean()
                )
            elif policy == "lbp_fresh":
                th_clean = float(
                    ((p_a.max(1) > theta) != (p_b.max(1) > theta))[clean].mean()
                )
            else:  # harp
                th_clean = float(
                    ((p_a.max(1) > theta) != (p_b.max(1) > theta))[clean].mean()
                )

            # Shared content cache under Zipf
            rng = np.random.RandomState(seed + 7)
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
                elif policy == "lbp_global":
                    row = p_global[v]
                elif policy == "lbp_seeded":
                    row = p_seeded[v]
                else:
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
                "raw_exactfrac": raw_ef,
                "replay_frac": replay,
                "cross_client_exactfrac": cross,
                "presession_audit_replay": audit_replay,
                "clean_threshold_flicker": th_clean,
                "shared_cache_hit": hit_rate,
                "n_clients": n_clients, "n_req": n_req, "cache_cap": cache_cap,
            })
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    df = pd.DataFrame(rows)
    cols = ["raw_exactfrac", "replay_frac", "cross_client_exactfrac",
            "presession_audit_replay", "clean_threshold_flicker", "shared_cache_hit"]
    means = df.groupby("policy")[cols].agg(["mean", "std"])
    print(means.round(4), flush=True)
    means.to_csv(os.path.join(OUT, "harp_exactfrac_sla_evidence_means.csv"))
    print("SLAEV DONE", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PRIVACYGNN_CONFIG", "experiment_config_confirmatory.yaml")
    main()
