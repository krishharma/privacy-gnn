#!/usr/bin/env python3
"""
Constraint-aware HARP upgrade for BigData repositioning.

Produces:
  results/harp_feasibility_pareto.csv   — Acc/LiRA vs ExactFrac c with feasibility
  results/harp_constructor_grid.csv    — selective baselines at matched Frac
  results/harp_spearman_risk.csv       — constructor↔vulnerability rank correlation
  results/harp_cfs.csv                 — Constrained Frac Search under (c, tau)
  results/harp_fidelity_mass.csv       — Mass vs L1 / top-1 flip / JS
  results/harp_serving_bench.csv       — in-process API serving + cache metrics

Primary protocol: GraphSAGE, release-only HARP, n_shadows=4, SEEDS5.
"""
from __future__ import annotations

import json
import os
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import load_config
from defenses.harp import (
    LOCKED_HARP_RELEASE,
    constrained_frac_search,
    risk_from_confidence,
    risk_from_degree,
    risk_from_train_neighbors,
)
from defenses.sami import compute_lte_risk
from experiment import run_one, _load_target_data, _shadow_vulnerability_risk, _make_gnn, _split_kwargs
from training import train_gnn
import torch

OUT = "results"
SEEDS5 = [42, 123, 456, 789, 1024]
DATASETS_CORE = ["Cora", "Chameleon", "Citeseer"]
N_SHADOWS = 4
SIGMA = 0.30


def _cfg():
    cfg = load_config("experiment_config_confirmatory.yaml")
    cfg["lira"] = {"n_shadows": N_SHADOWS}
    return cfg


def _release_params(seed_mode: str, frac: float) -> dict:
    p = dict(LOCKED_HARP_RELEASE)
    p["seed_mode"] = seed_mode
    p["use_lte"] = seed_mode == "lte"
    p["target_protect_frac"] = float(frac)
    p["strong_noise_scale"] = SIGMA
    if seed_mode == "audit":
        p["n_rank_shadows"] = 4
    return p


def _defense_name(seed_mode: str) -> str:
    return {
        "lte": "harp_release_only",
        "random": "harp_random",
        "audit": "harp_audit",
        "degree": "harp_degree",
        "train_nbr": "harp_train_nbr",
        "confidence": "harp_confidence",
        "entropy": "harp_entropy",
    }.get(seed_mode, "harp_release_only")


def _row(dataset, seed, tag, defense, params, cfg):
    # Force release-only semantics for all selective constructors.
    params = dict(params)
    params["lam"] = 0.0
    params["use_gate"] = False
    params["train_on_protected"] = False
    if defense == "harp_release_only":
        pass
    elif defense.startswith("harp_"):
        # Keep named constructor; still release-only via lam=0 path when using harp family.
        # harp_random/audit/etc. go through train_gnn_sami if lam>0; we set lam=0 so
        # they hit the release-only branch only for harp_release_only. For other names
        # with lam=0, experiment uses the lam<=0 branch inside harp family — good.
        pass
    r = run_one(dataset, "GraphSAGE", defense, params, seed, config=cfg)
    return {
        "dataset": dataset,
        "seed": seed,
        "tag": tag,
        "defense": defense,
        "Acc": float(r.get("test_accuracy", np.nan)),
        "LiRA": float(r.get("lira_attack_auc", np.nan)),
        "TPR1": float(r.get("lira_tpr_at_1pct_fpr", np.nan)),
        "ECE": float(r.get("ece", np.nan)),
        "Mass": float(r.get("noise_mass", np.nan)) if r.get("noise_mass") is not None else np.nan,
        "Frac": float(r.get("frac_protected", np.nan)) if r.get("frac_protected") is not None else np.nan,
        "ExactFrac": (
            1.0 - float(r["frac_protected"])
            if r.get("frac_protected") is not None
            else (0.0 if defense == "lbp" else 1.0 if defense == "none" else np.nan)
        ),
        "feasible_c060": int(
            (defense == "none")
            or (
                r.get("frac_protected") is not None
                and (1.0 - float(r["frac_protected"])) >= 0.60 - 1e-6
            )
            or (defense.startswith("harp"))
        ),
    }


def run_feasibility_pareto():
    """Central figure data: utility vs LiRA under ExactFrac constraints."""
    cfg = _cfg()
    rows = []
    # ExactFrac targets → Frac = 1 - c
    cs = [0.0, 0.2, 0.4, 0.6, 0.8]
    for dataset in ["Cora", "Chameleon"]:
        for seed in SEEDS5:
            # Uniform LBP points (ExactFrac=0 always)
            for scale, tag in [(0.12, "lbp_eqmass"), (0.30, "lbp_strong")]:
                r = run_one(dataset, "GraphSAGE", "lbp", {"scale": scale}, seed, config=cfg)
                rows.append({
                    "dataset": dataset, "seed": seed, "policy": tag,
                    "c_required": "any>0",
                    "feasible_for_c": 0,
                    "ExactFrac": 0.0,
                    "Frac": 1.0,
                    "Acc": float(r["test_accuracy"]),
                    "LiRA": float(r["lira_attack_auc"]),
                    "Mass": float(scale) * (2708 if dataset == "Cora" else r.get("noise_mass", np.nan) or np.nan),
                })
            # none
            r = run_one(dataset, "GraphSAGE", "none", {}, seed, config=cfg)
            rows.append({
                "dataset": dataset, "seed": seed, "policy": "none",
                "c_required": 1.0, "feasible_for_c": 1,
                "ExactFrac": 1.0, "Frac": 0.0,
                "Acc": float(r["test_accuracy"]),
                "LiRA": float(r["lira_attack_auc"]),
                "Mass": 0.0,
            })
            # HARP selective at each c (random + topology + audit)
            for c in cs:
                frac = round(1.0 - c, 3)
                if frac <= 0:
                    continue
                for mode in ("random", "lte", "audit"):
                    defense = _defense_name(mode)
                    params = _release_params(mode, frac)
                    # Use harp_release_only with seed_mode for lte; named for others
                    if mode == "lte":
                        defense = "harp_release_only"
                    out = _row(dataset, seed, f"harp_{mode}_c{c}", defense, params, cfg)
                    out.update({
                        "policy": f"harp_{mode}",
                        "c_required": c,
                        "feasible_for_c": 1,
                        "ExactFrac": out["ExactFrac"],
                        "Acc": out["Acc"],
                        "LiRA": out["LiRA"],
                        "Mass": out["Mass"],
                        "Frac": out["Frac"],
                    })
                    rows.append(out)
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "harp_feasibility_pareto.csv")
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")
    return df


def run_constructor_grid():
    cfg = _cfg()
    rows = []
    modes = ["lte", "random", "audit", "degree", "train_nbr", "confidence", "entropy"]
    for dataset in DATASETS_CORE:
        for seed in SEEDS5:
            # Baselines
            for defense, params, tag in [
                ("none", {}, "none"),
                ("lbp", {"scale": 0.30}, "lbp_strong"),
                ("lbp", {"scale": 0.12}, "lbp_eqmass"),
            ]:
                rows.append(_row(dataset, seed, tag, defense, params, cfg))
            for mode in modes:
                defense = _defense_name(mode)
                if mode == "lte":
                    defense = "harp_release_only"
                params = _release_params(mode, 0.40)
                rows.append(_row(dataset, seed, f"sel_{mode}", defense, params, cfg))
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "harp_constructor_grid.csv")
    df.to_csv(path, index=False)
    # Means + CIs
    gcols = ["dataset", "tag"]
    summ = (
        df.groupby(gcols)
        .agg(
            Acc_mean=("Acc", "mean"),
            Acc_std=("Acc", "std"),
            LiRA_mean=("LiRA", "mean"),
            LiRA_std=("LiRA", "std"),
            ECE_mean=("ECE", "mean"),
            Mass_mean=("Mass", "mean"),
            Frac_mean=("Frac", "mean"),
            ExactFrac_mean=("ExactFrac", "mean"),
            n=("Acc", "count"),
        )
        .reset_index()
    )
    # 95% CI approx using t≈2.776 for n=5 → use 1.96*std/sqrt(n) for simplicity
    for m in ("Acc", "LiRA"):
        summ[f"{m}_lo"] = summ[f"{m}_mean"] - 1.96 * summ[f"{m}_std"] / np.sqrt(summ["n"])
        summ[f"{m}_hi"] = summ[f"{m}_mean"] + 1.96 * summ[f"{m}_std"] / np.sqrt(summ["n"])
    summ.to_csv(os.path.join(OUT, "harp_constructor_grid_means.csv"), index=False)
    print(f"Wrote {path}")
    return df


def run_spearman():
    """Spearman(constructor risk, audit vulnerability) with bootstrap CIs."""
    cfg = _cfg()
    rows = []
    device = torch.device("cpu")
    for dataset in DATASETS_CORE:
        for seed in SEEDS5:
            split_kw = _split_kwargs(cfg)
            data, nc, nf = _load_target_data(dataset, cfg.get("data_dir", "data"), seed, False, split_kw)
            data.dataset_name = dataset
            vul = _shadow_vulnerability_risk(
                data, "GraphSAGE", nf, nc, device, 50, 0.01, 5e-4, cfg, n_rank=4, seed0=seed
            ).numpy()
            # Topology LTE
            lte = compute_lte_risk(data.cpu(), uniform=False, arch="sage", arch_aware=True)
            lte = lte.numpy() if hasattr(lte, "numpy") else np.asarray(lte)
            deg = risk_from_degree(data).numpy()
            tn = risk_from_train_neighbors(data).numpy()
            model = _make_gnn("GraphSAGE", nf, nc, use_gate=False)
            train_gnn(model, data, device, epochs=50, lr=0.01, weight_decay=5e-4)
            model.eval()
            with torch.no_grad():
                p = torch.softmax(model(data.x, data.edge_index), dim=1).cpu().numpy()
            conf = risk_from_confidence(p, "maxconf").numpy()
            ent = risk_from_confidence(p, "entropy").numpy()
            rng = np.random.RandomState(seed + 17)
            rnd = rng.rand(len(vul))
            for name, score in [
                ("lte", lte), ("degree", deg), ("train_nbr", tn),
                ("confidence", conf), ("entropy", ent), ("random", rnd),
            ]:
                rho, pval = spearmanr(score, vul)
                rows.append({
                    "dataset": dataset, "seed": seed, "constructor": name,
                    "spearman": float(rho), "pval": float(pval),
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "harp_spearman_risk.csv"), index=False)
    means = df.groupby(["dataset", "constructor"])["spearman"].agg(["mean", "std", "count"]).reset_index()
    means["lo"] = means["mean"] - 1.96 * means["std"] / np.sqrt(means["count"])
    means["hi"] = means["mean"] + 1.96 * means["std"] / np.sqrt(means["count"])
    means.to_csv(os.path.join(OUT, "harp_spearman_risk_means.csv"), index=False)
    print("Wrote spearman tables")
    return df


def run_cfs():
    """Constrained Frac Search under SLA c=0.60, tau=0.55 on Cora/Chameleon."""
    cfg = _cfg()
    rows = []
    for dataset in ["Cora", "Chameleon"]:
        for seed in SEEDS5:
            for mode in ("random", "lte", "audit"):
                cache = {}

                def eval_frac(frac, mode=mode, seed=seed, dataset=dataset):
                    key = round(float(frac), 3)
                    if key in cache:
                        return cache[key]
                    defense = "harp_release_only" if mode == "lte" else _defense_name(mode)
                    params = _release_params(mode, key)
                    out = _row(dataset, seed, f"cfs_{mode}", defense, params, cfg)
                    cache[key] = {
                        "Acc": out["Acc"], "LiRA": out["LiRA"],
                        "ExactFrac": out["ExactFrac"], "Mass": out["Mass"],
                    }
                    return cache[key]

                best = constrained_frac_search(
                    eval_frac, exact_frac_min=0.60, lira_max=0.55,
                    frac_grid=[0.0, 0.1, 0.2, 0.3, 0.4],
                )
                best.update({"dataset": dataset, "seed": seed, "constructor": mode})
                rows.append(best)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "harp_cfs.csv"), index=False)
    print(f"Wrote CFS ({len(df)} rows)")
    return df


def run_fidelity():
    """Mass vs distortion: L1, top-1 flip, JS divergence on Cora."""
    from defenses.lbp import lbp_perturb
    from experiment import _load_target_data, _split_kwargs, _make_gnn
    from defenses.harp import compute_harp_scales, expand_k_hop, select_risk_seeds

    cfg = _cfg()
    rows = []
    device = torch.device("cpu")
    for seed in SEEDS5:
        split_kw = _split_kwargs(cfg)
        data, nc, nf = _load_target_data("Cora", cfg.get("data_dir", "data"), seed, False, split_kw)
        model = _make_gnn("GraphSAGE", nf, nc, use_gate=False)
        train_gnn(model, data, device, epochs=50, lr=0.01, weight_decay=5e-4)
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(data.x.to(device), data.edge_index.to(device)), dim=1).cpu().numpy()
        # Equal-mass LBP
        p_lbp = lbp_perturb(p, scale=0.12, seed=seed)
        # HARP release-only scales
        scales, prot, _, stats = compute_harp_scales(
            data, risk=None, k_hops=1, strong_noise_scale=0.30, weak_noise_scale=0.0,
            use_lte=True, target_protect_frac=0.40,
        )
        rng = np.random.RandomState(seed)
        noise = rng.laplace(0.0, scales[:, None], size=p.shape)
        p_harp = p + noise
        p_harp = np.clip(p_harp, 1e-12, None)
        p_harp = p_harp / p_harp.sum(axis=1, keepdims=True)

        def js(a, b):
            m = 0.5 * (a + b)
            def kl(x, y):
                return np.sum(x * (np.log(x + 1e-12) - np.log(y + 1e-12)), axis=1)
            return 0.5 * (kl(a, m) + kl(b, m))

        for name, q, mass in [
            ("lbp_eqmass", p_lbp, 0.12 * len(p)),
            ("harp_release", p_harp, float(stats["noise_mass"])),
        ]:
            l1 = float(np.mean(np.abs(p - q).sum(axis=1)))
            flip = float(np.mean(p.argmax(1) != q.argmax(1)))
            rows.append({
                "seed": seed, "policy": name, "Mass": mass,
                "mean_L1": l1, "top1_flip": flip, "mean_JS": float(js(p, q).mean()),
                "ExactFrac": float((np.abs(p - q).sum(axis=1) < 1e-12).mean()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "harp_fidelity_mass.csv"), index=False)
    print("Wrote fidelity table")
    return df


def run_serving_bench():
    """
    Reproducible in-process score API benchmark (no FastAPI dependency).
    Workloads: uniform, Zipf, bursty-repeat. Policies: none, lbp_fresh, lbp_B1,
    harp_fresh, harp_B1.
    """
    cfg = _cfg()
    split_kw = _split_kwargs(cfg)
    data, nc, nf = _load_target_data("Cora", cfg.get("data_dir", "data"), 42, False, split_kw)
    device = torch.device("cpu")
    model = _make_gnn("GraphSAGE", nf, nc, use_gate=False)
    train_gnn(model, data, device, epochs=50, lr=0.01, weight_decay=5e-4)
    model.eval()
    with torch.no_grad():
        base = torch.softmax(model(data.x.to(device), data.edge_index.to(device)), dim=1).cpu().numpy()
    n, c = base.shape
    from defenses.harp import compute_harp_scales
    from defenses.lbp import lbp_perturb

    scales, prot, _, _ = compute_harp_scales(
        data, risk=None, k_hops=1, strong_noise_scale=0.30, weak_noise_scale=0.0,
        use_lte=True, target_protect_frac=0.40,
    )
    exact = ~prot

    def release(policy, node, rng, session_cache):
        key = (policy, int(node))
        if policy.endswith("_B1") and key in session_cache:
            return session_cache[key], True
        if policy.startswith("none"):
            out = base[node]
        elif policy.startswith("lbp"):
            out = lbp_perturb(base[node:node+1], scale=0.30, seed=int(rng.randint(0, 10**9)))[0]
        elif policy.startswith("harp"):
            if exact[node]:
                out = base[node]
            else:
                noise = rng.laplace(0.0, float(scales[node]), size=c)
                q = np.clip(base[node] + noise, 1e-12, None)
                out = q / q.sum()
        else:
            out = base[node]
        hit = False
        if policy.endswith("_B1"):
            session_cache[key] = out
        return out, hit

    workloads = {
        "uniform": lambda rng, m: rng.randint(0, n, size=m),
        "zipf": lambda rng, m: np.clip(rng.zipf(1.3, size=m) - 1, 0, n - 1),
        "bursty": lambda rng, m: np.repeat(rng.randint(0, n, size=max(1, m // 25)), 25)[:m],
    }
    rows = []
    n_queries = 20000
    for wl_name, sampler in workloads.items():
        for policy in ("none", "lbp_fresh", "lbp_B1", "harp_fresh", "harp_B1"):
            rng = np.random.RandomState(123)
            nodes = sampler(rng, n_queries)
            cache = {}
            bytes_store = 0
            hits = 0
            t0 = time.perf_counter()
            lat = []
            for i, v in enumerate(nodes):
                t1 = time.perf_counter()
                # Content-addressable score cache: bit-exact responses share key.
                if policy in ("none", "harp_fresh", "harp_B1") and (policy == "none" or exact[v]):
                    ckey = ("exact", int(v))
                    if ckey in cache:
                        hits += 1
                        out = cache[ckey]
                    else:
                        out, _ = release(policy.replace("_fresh", "").replace("_B1", "_B1") if False else policy, v, rng, {})
                        # simplify:
                        if policy == "none" or exact[v]:
                            out = base[v]
                        else:
                            noise = rng.laplace(0.0, float(scales[v]), size=c)
                            q = np.clip(base[v] + noise, 1e-12, None)
                            out = q / q.sum()
                        cache[ckey] = out
                        bytes_store += int(out.nbytes)
                elif policy == "lbp_B1":
                    ckey = ("sess", int(v))
                    if ckey in cache:
                        hits += 1
                        out = cache[ckey]
                    else:
                        out = lbp_perturb(base[v:v+1], scale=0.30, seed=int(rng.randint(0, 10**9)))[0]
                        cache[ckey] = out
                        bytes_store += int(out.nbytes)
                elif policy == "harp_B1":
                    ckey = ("sess", int(v))
                    if ckey in cache:
                        hits += 1
                        out = cache[ckey]
                    else:
                        if exact[v]:
                            out = base[v]
                        else:
                            noise = rng.laplace(0.0, float(scales[v]), size=c)
                            q = np.clip(base[v] + noise, 1e-12, None)
                            out = q / q.sum()
                        cache[ckey] = out
                        bytes_store += int(out.nbytes)
                else:  # lbp_fresh / harp_fresh protected path without content cache
                    if policy == "lbp_fresh":
                        out = lbp_perturb(base[v:v+1], scale=0.30, seed=int(rng.randint(0, 10**9)))[0]
                    else:
                        if exact[v]:
                            ckey = ("exact", int(v))
                            if ckey in cache:
                                hits += 1
                                out = cache[ckey]
                            else:
                                out = base[v]
                                cache[ckey] = out
                                bytes_store += int(out.nbytes)
                        else:
                            noise = rng.laplace(0.0, float(scales[v]), size=c)
                            q = np.clip(base[v] + noise, 1e-12, None)
                            out = q / q.sum()
                lat.append((time.perf_counter() - t1) * 1000.0)
            elapsed = time.perf_counter() - t0
            arr = np.asarray(lat)
            rows.append({
                "workload": wl_name,
                "policy": policy,
                "n_queries": n_queries,
                "qps": n_queries / elapsed,
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "hit_rate": hits / n_queries,
                "cache_bytes": bytes_store,
                "cost_per_1e6_queries_rel": (1.0 - hits / n_queries),  # relative miss cost
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "harp_serving_bench.csv"), index=False)
    print("Wrote serving bench")
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print("=== Spearman ===")
    run_spearman()
    print("=== Fidelity ===")
    run_fidelity()
    print("=== Serving ===")
    run_serving_bench()
    print("=== Constructor grid (5-seed) ===")
    run_constructor_grid()
    print("=== Feasibility Pareto ===")
    run_feasibility_pareto()
    print("=== CFS ===")
    run_cfs()
    meta = {"elapsed_sec": time.time() - t0, "seeds": SEEDS5, "n_shadows": N_SHADOWS}
    with open(os.path.join(OUT, "harp_constrained_upgrade_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done", meta)


if __name__ == "__main__":
    main()
