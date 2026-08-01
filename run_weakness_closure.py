#!/usr/bin/env python3
"""
Close remaining HARP paper weaknesses with measured experiments.

W1 Scale: products BFS 40k n_sh=2; products 15k n_sh=8 (subset seeds)
W2 Unconstrained: HARP Frac=1 / harp_uniform vs strong LBP
W3 MemGuard+GAP remeasure (ExactFrac=1 for GAP)
W4 Slice-aware CFS
W5 Ensemble constructor vs random/lte/audit
W6 Sybil: per-principal B vs shared global budget
W7 5-seed CFS + inductive
W8 Hybrid DP already via GAP ExactFrac=1
W9 Label-only + edge MIA under defenses
"""
from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from config import load_config
from defenses.harp import (
    LOCKED_HARP_RELEASE,
    constrained_frac_search,
    slice_constrained_frac_search,
)
from defenses.sami import risk_scaled_posterior_noise
from experiment import run_one, _train_and_predict_gnn, _load_target_data, _split_kwargs
from attacks import average_posterior_queries

OUT = "results"
SEEDS5 = [42, 123, 456, 789, 1024]


def _cfg(n_sh=4):
    cfg = load_config("experiment_config_confirmatory.yaml")
    cfg["lira"] = {"n_shadows": int(n_sh)}
    return cfg


def _append(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.isfile(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=[c for c in df.columns if c in ("tag", "seed", "policy", "Frac", "B", "n_sub", "n_shadows", "experiment", "constructor", "query_set")], keep="last")
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# W2 + W3 + W5 + W8 + W9: core Cora grid
# ---------------------------------------------------------------------------
def run_core_grid():
    cfg = _cfg(4)
    path = os.path.join(OUT, "harp_weakness_core.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.isfile(path) else []
    done = {(r["tag"], int(r["seed"])) for r in rows}
    jobs = [
        ("none", "none", {}),
        ("lbp_eq", "lbp", {"scale": 0.12}),
        ("lbp_strong", "lbp", {"scale": 0.30}),
        ("harp_locked", "harp_release_only", dict(LOCKED_HARP_RELEASE)),
        ("harp_full", "harp_uniform", {**LOCKED_HARP_RELEASE, "target_protect_frac": 1.0, "risk_frac": 1.0, "k_hops": 0, "strong_noise_scale": 0.30}),
        ("harp_random", "harp_random", {**LOCKED_HARP_RELEASE, "seed_mode": "random"}),
        ("harp_audit", "harp_audit", {**LOCKED_HARP_RELEASE, "seed_mode": "audit"}),
        ("harp_ensemble", "harp_ensemble", {**LOCKED_HARP_RELEASE, "seed_mode": "ensemble"}),
        ("harp_entropy", "harp_entropy", {**LOCKED_HARP_RELEASE, "seed_mode": "entropy"}),
        ("memguard", "memguard", {"max_l1": 0.2, "n_steps": 30}),
        ("gap_s3", "gap_agg", {"sigma": 3.0, "epochs": 80, "max_degree": 100}),
        ("gap_s5", "gap_agg", {"sigma": 5.0, "epochs": 80, "max_degree": 100}),
    ]
    for seed in SEEDS5:
        for tag, dn, dp in jobs:
            if (tag, seed) in done:
                continue
            print(f"CORE {tag} seed={seed}", flush=True)
            r = run_one("Cora", "GraphSAGE", dn, dp, seed, config=cfg)
            rows.append({
                "tag": tag, "seed": seed,
                "Acc": float(r["test_accuracy"]),
                "LiRA": float(r["lira_attack_auc"]),
                "ECE": float(r.get("ece_test", np.nan)),
                "ExactFrac": float(r.get("exact_frac", np.nan)),
                "Mass": r.get("noise_mass"),
                "Frac": r.get("frac_protected"),
                "eps": r.get("dp_epsilon"),
                "edge_mia": r.get("edge_mia_auc"),
                "label_gap": r.get("label_only_gap_auc"),
            })
            done.add((tag, seed))
            pd.DataFrame(rows).to_csv(path, index=False)
            print(
                f"  Acc={rows[-1]['Acc']:.3f} LiRA={rows[-1]['LiRA']:.3f} "
                f"EF={rows[-1]['ExactFrac']:.3f} edge={rows[-1]['edge_mia']} eps={rows[-1]['eps']}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    means = df.groupby("tag").agg(
        Acc=("Acc", "mean"), LiRA=("LiRA", "mean"), ECE=("ECE", "mean"),
        ExactFrac=("ExactFrac", "mean"), eps=("eps", "mean"),
        edge_mia=("edge_mia", "mean"), label_gap=("label_gap", "mean"),
        Acc_std=("Acc", "std"), LiRA_std=("LiRA", "std"), n=("Acc", "count"),
    ).reset_index()
    means.to_csv(os.path.join(OUT, "harp_weakness_core_means.csv"), index=False)
    print(means.to_string(), flush=True)
    return df


# ---------------------------------------------------------------------------
# W4: slice-aware CFS
# ---------------------------------------------------------------------------
def _clean_slice_auc(p, yn, trm, tem, prot):
    unprot = ~np.asarray(prot, dtype=bool)
    m = trm & unprot
    n = tem & unprot
    if m.sum() < 5 or n.sum() < 5:
        return float("nan")
    conf = p[np.arange(len(yn)), yn]
    s = np.concatenate([conf[m], conf[n]])
    y = np.concatenate([np.ones(int(m.sum())), np.zeros(int(n.sum()))])
    return float(roc_auc_score(y, s))


def run_slice_cfs():
    from defenses.harp import compute_harp_scales
    cfg = _cfg(4)
    path = os.path.join(OUT, "harp_slice_cfs.csv")
    rows = []
    device = torch.device(cfg.get("device", "cpu"))
    split_kw = _split_kwargs(cfg)
    ep = int(cfg.get("training", {}).get("epochs", 50))
    lr = float(cfg.get("training", {}).get("lr", 0.01))
    wd = float(cfg.get("training", {}).get("weight_decay", 5e-4))
    for seed in SEEDS5:
        print(f"SLICE-CFS seed={seed}", flush=True)
        cache = {}

        def eval_frac(frac, seed=seed):
            key = round(float(frac), 3)
            if key in cache:
                return cache[key]
            params = {**LOCKED_HARP_RELEASE, "target_protect_frac": key, "seed_mode": "ensemble"}
            r = run_one("Cora", "GraphSAGE", "harp_ensemble", params, seed, config=cfg)
            # Approximate slice AUC from a fresh release with known protect mask
            data, nc, nf = _load_target_data("Cora", cfg["data_dir"], seed, True, split_kw)
            p, pr, risk, _, _, rel = _train_and_predict_gnn(
                "GraphSAGE", "harp_ensemble", params, data, nf, nc, device,
                ep, lr, wd, {}, None, False, 1024, [15, 10], cfg, release_seed=seed,
            )
            scales, prot, _, _ = compute_harp_scales(
                data.cpu(), risk=risk, risk_frac=params["risk_frac"], k_hops=params["k_hops"],
                strong_noise_scale=params["strong_noise_scale"], weak_noise_scale=0.0,
                target_protect_frac=key, arch="sage", arch_aware=True,
            )
            yn = data.y.numpy(); trm = data.train_mask.numpy(); tem = data.test_mask.numpy()
            sa = _clean_slice_auc(p, yn, trm, tem, prot)
            out = {
                "Acc": float(r["test_accuracy"]),
                "LiRA": float(r["lira_attack_auc"]),
                "ExactFrac": float(r.get("exact_frac", 1.0 - key)),
                "Mass": r.get("noise_mass"),
                "slice_auc": float(sa) if sa == sa else 1.0,
            }
            cache[key] = out
            print(f"  frac={key} Acc={out['Acc']:.3f} LiRA={out['LiRA']:.3f} slice={out['slice_auc']:.3f}", flush=True)
            return out

        best = slice_constrained_frac_search(
            eval_frac, exact_frac_min=0.60, slice_auc_max=0.58, lira_max=0.55,
            frac_grid=[0.0, 0.2, 0.3, 0.4],
        )
        best.update({"dataset": "Cora", "seed": seed})
        rows.append(best)
        print("BEST", best, flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    print(pd.DataFrame(rows).mean(numeric_only=True), flush=True)
    return rows


# ---------------------------------------------------------------------------
# W6: Sybil / shared budget
# ---------------------------------------------------------------------------
def run_sybil_budget():
    """Compare per-principal B=1 vs shared global draw pool across N fake identities."""
    from defenses.harp import compute_harp_scales
    cfg = _cfg(4)
    path = os.path.join(OUT, "harp_sybil_budget.csv")
    rows = []
    device = torch.device(cfg.get("device", "cpu"))
    split_kw = _split_kwargs(cfg)
    ep = int(cfg.get("training", {}).get("epochs", 50))
    lr = float(cfg.get("training", {}).get("lr", 0.01))
    wd = float(cfg.get("training", {}).get("weight_decay", 5e-4))
    K = 20
    for seed in SEEDS5:
        print(f"SYBIL seed={seed}", flush=True)
        data, nc, nf = _load_target_data("Cora", cfg["data_dir"], seed, True, split_kw)
        params = {**LOCKED_HARP_RELEASE, "strong_noise_scale": 0.0}
        p_base, _, risk, _, _, _ = _train_and_predict_gnn(
            "GraphSAGE", "harp_release_only", params, data, nf, nc, device,
            ep, lr, wd, {}, None, False, 1024, [15, 10], cfg, release_seed=seed,
        )
        scales, prot, _, _ = compute_harp_scales(
            data.cpu(), risk=risk, risk_frac=LOCKED_HARP_RELEASE["risk_frac"],
            k_hops=1, strong_noise_scale=0.30, weak_noise_scale=0.0,
            target_protect_frac=0.40, arch="sage", arch_aware=True,
        )
        risk_bin = (np.asarray(scales) > 1e-12).astype(float)
        strong = 0.30
        yn = data.y.numpy(); tem = data.test_mask.numpy()

        def acc_of(p):
            return float((p.argmax(1)[tem] == yn[tem]).mean())

        # Per-principal B=1: each of N identities gets 1 fresh draw, then averages K queries with reuse
        for n_id in (1, 5, 20):
            # Sybil abuse of per-principal B=1: N identities × 1 draw = N unique draws averaged
            draws = [risk_scaled_posterior_noise(p_base, risk_bin, strong, seed=seed + i) for i in range(n_id)]
            while len(draws) < K:
                draws.append(draws[len(draws) % n_id])
            p_sybil = np.mean(draws[:K], axis=0)
            # Shared global budget B_global=1: all identities share one draw
            shared = [risk_scaled_posterior_noise(p_base, risk_bin, strong, seed=seed + 999)]
            while len(shared) < K:
                shared.append(shared[0])
            p_shared = np.mean(shared[:K], axis=0)
            # Uncapped reference
            p_unc = average_posterior_queries(p_base, risk_bin, strong, K, seed0=seed)
            rows.append({
                "seed": seed, "n_identities": n_id,
                "acc_sybil_per_principal": acc_of(p_sybil),
                "acc_shared_global_B1": acc_of(p_shared),
                "acc_uncapped": acc_of(p_unc),
                "acc_oneshot": acc_of(risk_scaled_posterior_noise(p_base, risk_bin, strong, seed=seed)),
            })
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    print(pd.DataFrame(rows).groupby("n_identities").mean(numeric_only=True).round(4), flush=True)
    return rows


# ---------------------------------------------------------------------------
# W7: CFS + inductive 5-seed
# ---------------------------------------------------------------------------
def run_cfs_5seed():
    cfg = _cfg(4)
    path = os.path.join(OUT, "harp_cfs.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.isfile(path) else []
    done = {(int(r["seed"]), r["constructor"]) for r in rows if "constructor" in r}
    for seed in SEEDS5:
        for mode in ("random", "lte", "audit", "ensemble"):
            if (seed, mode) in done:
                continue
            print(f"CFS {mode} seed={seed}", flush=True)
            cache = {}

            def eval_frac(frac, mode=mode, seed=seed):
                key = round(float(frac), 3)
                if key in cache:
                    return cache[key]
                if mode == "lte":
                    dn, params = "harp_release_only", {**LOCKED_HARP_RELEASE, "target_protect_frac": key}
                elif mode == "ensemble":
                    dn, params = "harp_ensemble", {**LOCKED_HARP_RELEASE, "target_protect_frac": key, "seed_mode": "ensemble"}
                else:
                    dn = f"harp_{mode}"
                    params = {**LOCKED_HARP_RELEASE, "target_protect_frac": key, "seed_mode": mode}
                r = run_one("Cora", "GraphSAGE", dn, params, seed, config=cfg)
                out = {
                    "Acc": float(r["test_accuracy"]),
                    "LiRA": float(r["lira_attack_auc"]),
                    "ExactFrac": float(r.get("exact_frac", 1.0 - key)),
                    "Mass": r.get("noise_mass"),
                }
                cache[key] = out
                print(f"  frac={key} Acc={out['Acc']:.3f} LiRA={out['LiRA']:.3f}", flush=True)
                return out

            best = constrained_frac_search(eval_frac, exact_frac_min=0.60, lira_max=0.55, frac_grid=[0.0, 0.2, 0.3, 0.4])
            best.update({"dataset": "Cora", "seed": seed, "constructor": mode})
            rows.append(best)
            done.add((seed, mode))
            pd.DataFrame(rows).to_csv(path, index=False)
            print("BEST", best, flush=True)
    return rows


def run_inductive_5seed():
    cfg = _cfg(4)
    path = os.path.join(OUT, "harp_inductive_5seed.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.isfile(path) else []
    done = {(r["tag"], int(r["seed"])) for r in rows}
    # Reuse competitiveness inductive helper if available
    try:
        from run_harp_competitiveness_upgrade import inductive_exclude_test_edges
    except Exception:
        inductive_exclude_test_edges = None
    split_kw = _split_kwargs(cfg)
    device = torch.device(cfg.get("device", "cpu"))
    ep = int(cfg.get("training", {}).get("epochs", 50))
    lr = float(cfg.get("training", {}).get("lr", 0.01))
    wd = float(cfg.get("training", {}).get("weight_decay", 5e-4))
    for seed in SEEDS5:
        for tag, dn, dp in [
            ("none", "none", {}),
            ("lbp_strong", "lbp", {"scale": 0.3}),
            ("harp", "harp_release_only", dict(LOCKED_HARP_RELEASE)),
        ]:
            if (tag, seed) in done:
                continue
            print(f"INDUCTIVE {tag} seed={seed}", flush=True)
            data, nc, nf = _load_target_data("Cora", cfg["data_dir"], seed, True, split_kw)
            if inductive_exclude_test_edges is not None:
                data = inductive_exclude_test_edges(data)
            else:
                # Drop edges incident to test nodes
                ei = data.edge_index
                tem = data.test_mask
                keep = ~(tem[ei[0]] | tem[ei[1]])
                data = data.clone()
                data.edge_index = ei[:, keep]
            # Use run_one path via temporary: call _train + LiRA manually through run_one on modified?
            # Simpler: run_one doesn't accept custom data — use local LiRA like products script
            from lira_attack import _logit_confidence
            from sklearn.metrics import roc_auc_score as auc
            p, pr, _, _, _, rel = _train_and_predict_gnn(
                "GraphSAGE", dn, dp, data, nf, nc, device, ep, lr, wd, {}, None, False,
                1024, [15, 10], cfg, release_seed=seed,
            )
            yn = data.y.view(-1).cpu().numpy()
            m = data.num_nodes
            n_tr = int(data.train_mask.sum())
            conf_t = _logit_confidence(p, yn)
            in_mu = np.zeros(m); out_mu = np.zeros(m); in_n = np.zeros(m); out_n = np.zeros(m)
            for k in range(4):
                sdata = data.clone()
                rng2 = np.random.RandomState(seed + 1000 + k)
                perm2 = rng2.permutation(m)
                tr2 = torch.zeros(m, dtype=torch.bool)
                tr2[perm2[:n_tr]] = True
                sdata.train_mask = tr2
                sp, _, _, _, _, _ = _train_and_predict_gnn(
                    "GraphSAGE", dn, dp, sdata, nf, nc, device, ep, lr, wd, {}, None, False,
                    1024, [15, 10], cfg, release_seed=seed + k,
                )
                conf = _logit_confidence(sp, yn)
                sm = tr2.cpu().numpy()
                in_mu += conf * sm; in_n += sm
                out_mu += conf * (~sm); out_n += (~sm)
            in_mu /= np.maximum(in_n, 1); out_mu /= np.maximum(out_n, 1)
            score = -np.abs(conf_t - in_mu) + np.abs(conf_t - out_mu)
            trn = data.train_mask.cpu().numpy(); ten = data.test_mask.cpu().numpy()
            mask = trn | ten
            la = float(auc(trn[mask].astype(int), score[mask]))
            acc = float((pr[ten] == yn[ten]).mean())
            rows.append({"tag": tag, "seed": seed, "Acc": acc, "LiRA": la,
                         "ExactFrac": 1.0 - float(rel["frac_protected"]) if rel.get("frac_protected") is not None else (0.0 if tag.startswith("lbp") else 1.0)})
            done.add((tag, seed))
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  Acc={acc:.3f} LiRA={la:.3f}", flush=True)
    means = pd.DataFrame(rows).groupby("tag")[["Acc", "LiRA"]].agg(["mean", "std", "count"])
    means.to_csv(os.path.join(OUT, "harp_inductive_5seed_means.csv"))
    print(means, flush=True)
    return rows


# ---------------------------------------------------------------------------
# W1: products scale
# ---------------------------------------------------------------------------
def run_products_scale():
    print("=== PRODUCTS 15k n_sh=8 (3 seeds) ===", flush=True)
    import run_products_nsh4 as rp
    try:
        rp.main(n_sub=15000, n_shadows=8, seeds=[42, 123, 456])
    except Exception as e:
        print("products nsh8 failed", e, flush=True)
    print("=== PRODUCTS 40k n_sh=2 (3 seeds) ===", flush=True)
    sub = os.path.join(OUT, "products_sub_40000.pt")
    if not os.path.isfile(sub):
        try:
            from run_harp_lock_gaps import build_products_bfs
            cfg = load_config("experiment_config_ogbn.yaml")
            root = os.path.join(cfg["data_dir"], "ogb")
            if not os.path.isdir(os.path.join(root, "ogbn_products", "processed")):
                root = cfg["data_dir"]
            build_products_bfs(40000, root, sub)
        except Exception as e:
            print("build 40k failed", e, flush=True)
            return
    try:
        rp.main(n_sub=40000, n_shadows=2, seeds=[42, 123, 456])
    except Exception as e:
        print("products 40k failed", e, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print("=== CORE GRID ===", flush=True)
    run_core_grid()
    print("=== SYBIL ===", flush=True)
    run_sybil_budget()
    print("=== CFS 5SEED ===", flush=True)
    run_cfs_5seed()
    print("=== INDUCTIVE 5SEED ===", flush=True)
    run_inductive_5seed()
    print("=== SLICE CFS ===", flush=True)
    try:
        run_slice_cfs()
    except Exception as e:
        print("slice cfs failed", e, flush=True)
    print("=== PRODUCTS SCALE ===", flush=True)
    try:
        run_products_scale()
    except Exception as e:
        print("products scale failed", e, flush=True)
    print("ALL WEAKNESS CLOSURE DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PRIVACYGNN_CONFIG", "experiment_config_confirmatory.yaml")
    main()
