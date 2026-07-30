"""
HARP evaluation suite (paper tables and stress checks).

Runs (resume-safe, writes incrementally under results/):
  1) Shadow-count completion (Chameleon n=64) + comparative figure
  2) Cache/veracity simulation + figure
  3) Hop-leakage synthetic validation
  4) Selective local-ε accounting
  5) Qualitative failure cases on Chameleon
  6) ogbn-arxiv LiRA with n_shadows=4
  7) ogbn-products subsampled defense-aware LiRA
"""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Dict, List, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from config import ensure_dirs, load_config
from defenses.harp import LOCKED_HARP, compute_harp_scales
from experiment import (
    MINIBATCH_DATASETS,
    _load_target_data,
    _make_shadow_data,
    _split_kwargs,
    _train_and_predict_gnn,
    run_one,
)
from lira_attack import lira_gaussian_auc

SEEDS3 = [42, 123, 456]
OUT = "results"
FIG = "paper/paper_visuals"


def cfg(name="experiment_config_confirmatory.yaml"):
    os.environ["PRIVACYGNN_CONFIG"] = name
    c = dict(load_config(name))
    ensure_dirs(c)
    c["attacks"] = ["confidence", "lira"]
    return c


def done_keys(path: str, cols: Tuple[str, ...]) -> Set[Tuple]:
    if not os.path.isfile(path):
        return set()
    df = pd.read_csv(path)
    keys = set()
    for _, r in df.iterrows():
        try:
            keys.add(tuple(r[c] for c in cols))
        except Exception:
            continue
    return keys


def append_csv(path: str, rows: List[Dict]):
    df = pd.DataFrame(rows)
    if os.path.isfile(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def ensure_dirs_fig():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs("figures", exist_ok=True)


def savefig(fig, name: str):
    ensure_dirs_fig()
    for root in [FIG, "figures"]:
        fig.savefig(f"{root}/{name}", dpi=200, bbox_inches="tight")
    print(f"wrote {name}", flush=True)


# ---------------------------------------------------------------------------
# Offline / fast
# ---------------------------------------------------------------------------
def run_local_eps_accounting():
    delta_l1 = 2.0
    rows = []
    for sigma in [0.10, 0.15, 0.30, 0.50]:
        eps = delta_l1 / sigma
        for frac in [0.0, 0.20, 0.40, 0.60, 1.0]:
            rows.append(
                {
                    "sigma": sigma,
                    "frac": frac,
                    "exact_frac": round(1.0 - frac, 4),
                    "eps_protected_local": round(eps, 4) if frac > 0 else None,
                    "global_eps": "inf" if frac < 1.0 else round(eps, 4),
                    "mass_cora": round(frac * 2708 * sigma, 1),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/harp_local_eps_accounting.csv", index=False)
    summary = {
        "delta_l1": delta_l1,
        "locked_sigma": 0.30,
        "locked_frac": 0.40,
        "eps_protected_local": round(delta_l1 / 0.30, 4),
        "exact_frac": 0.60,
        "global_eps": "infinity whenever ExactFrac>0",
        "interpretation": (
            "Finite local ε on the protected minority; ε=∞ on the clean majority. "
            "Global DP and ExactFrac>0 are incompatible by construction — the "
            "constrained-release tradeoff, not a missing proof."
        ),
    }
    with open(f"{OUT}/harp_local_eps_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return df


def run_cache_simulation(seed: int = 0):
    rng = np.random.RandomState(seed)
    n_nodes, n_classes, n_clients, n_queries = 2708, 7, 800, 25
    clean = rng.dirichlet(np.ones(n_classes), size=n_nodes)
    frac, sigma = 0.40, 0.30
    protected = rng.rand(n_nodes) < frac

    def release(policy, v, store):
        key = (policy, v)
        if policy == "none":
            hit = key in store
            store[key] = clean[v]
            return clean[v], hit
        def _laplace_renorm(base, b):
            p = np.clip(base + rng.laplace(0.0, b, size=n_classes), 1e-12, None)
            return p / p.sum()

        if policy == "lbp_fresh":
            return _laplace_renorm(clean[v], 0.3), False
        if policy == "lbp_B1":
            if key in store:
                return store[key], True
            p = _laplace_renorm(clean[v], 0.3)
            store[key] = p
            return p, False
        if policy == "harp_fresh":
            if not protected[v]:
                hit = key in store
                store[key] = clean[v]
                return clean[v], hit
            return _laplace_renorm(clean[v], sigma), False
        if policy == "harp_B1":
            if not protected[v]:
                hit = key in store
                store[key] = clean[v]
                return clean[v], hit
            if key in store:
                return store[key], True
            p = _laplace_renorm(clean[v], sigma)
            store[key] = p
            return p, False
        raise ValueError(policy)

    rows = []
    client_nodes = rng.randint(0, n_nodes, size=(n_clients, n_queries))
    for policy in ["none", "lbp_fresh", "lbp_B1", "harp_fresh", "harp_B1"]:
        store: Dict = {}
        hits = misses = 0
        stable = total_rep = 0
        prev = {}
        for c in range(n_clients):
            for q in range(n_queries):
                v = int(client_nodes[c, q])
                p, hit = release(policy, v, store)
                hits += int(hit)
                misses += int(not hit)
                ck = (c, v)
                if ck in prev:
                    total_rep += 1
                    if np.allclose(p, prev[ck]):
                        stable += 1
                prev[ck] = p
        total = hits + misses
        rows.append(
            {
                "policy": policy,
                "hit_rate": hits / total,
                "miss_rate": misses / total,
                "bitexact_requery_rate": (stable / total_rep) if total_rep else 1.0,
                "design_exact_frac": 1.0
                if policy == "none"
                else (0.0 if policy.startswith("lbp") else 1.0 - frac),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/harp_cache_simulation.csv", index=False)
    print(df.to_string(index=False), flush=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    labels = {
        "none": "None",
        "lbp_fresh": "LBP fresh",
        "lbp_B1": "LBP B=1",
        "harp_fresh": "HARP fresh",
        "harp_B1": "HARP B=1",
    }
    x = np.arange(len(df))
    ax.bar(x - 0.18, df.hit_rate, 0.36, label="Cache hit rate", color="#4c72b0")
    ax.bar(
        x + 0.18,
        df.bitexact_requery_rate,
        0.36,
        label="Bit-exact re-query rate",
        color="#55a868",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[p] for p in df.policy], rotation=18, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rate")
    ax.set_title("Veracity: score-cache behavior under re-query traffic (Cora-sized)")
    for i, row in df.iterrows():
        ax.text(i, 1.04, f"c={row.design_exact_frac:.0%}", ha="center", fontsize=7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig_harp_cache_veracity.png")
    plt.close()
    return df


def run_hop_leakage_synth(seed: int = 0):
    """
    Message-passing mixes a seed's membership cue into neighbor posteriors at
    inference time *before* release noise. Noising only the seed at release
    leaves the clean neighbor score intact — so hop expansion is necessary.
    """
    rng = np.random.RandomState(seed)
    trials = 400
    rows = []
    for _ in range(trials):
        mem = int(rng.rand() < 0.5)
        # Clean posteriors after one-hop mixing (fixed before release):
        # leaf confidence already carries the seed membership cue.
        leaf_clean = 0.55 + (0.35 if mem else -0.35) + rng.randn() * 0.05
        seed_clean = 0.60 + (0.30 if mem else -0.30) + rng.randn() * 0.05
        for policy in ["unprotected", "seed_only", "hop1", "uniform"]:
            seed_rel = seed_clean
            leaf_rel = leaf_clean
            if policy in ("seed_only", "hop1", "uniform"):
                seed_rel = seed_clean + rng.laplace(0.0, 0.6)
            if policy in ("hop1", "uniform"):
                leaf_rel = leaf_clean + rng.laplace(0.0, 0.6)
            # Attacker queries the leaf (natural API client behavior)
            rows.append({"policy": policy, "mem": mem, "feat": float(leaf_rel)})
    df = pd.DataFrame(rows)
    aucs = []
    for policy, g in df.groupby("policy"):
        aucs.append(
            {
                "policy": policy,
                "attack_auc": round(float(roc_auc_score(g["mem"], g["feat"])), 4),
            }
        )
    out = pd.DataFrame(aucs)
    out.to_csv(f"{OUT}/harp_hop_leakage_synth.csv", index=False)
    print(out.to_string(index=False), flush=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    order = ["unprotected", "seed_only", "hop1", "uniform"]
    vals = [float(out.loc[out.policy == p, "attack_auc"].iloc[0]) for p in order]
    colors = ["#444444", "#c44e52", "#4c72b0", "#8172b3"]
    ax.bar(order, vals, color=colors)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Membership AUROC via leaf scores")
    ax.set_title("Hop-consistency necessity (synthetic one-hop mixer)")
    ax.set_ylim(0.4, 1.05)
    fig.tight_layout()
    savefig(fig, "fig_harp_hop_necessity.png")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Shadow completion + figure
# ---------------------------------------------------------------------------
def run_shadow_completion(device, c):
    path = f"{OUT}/harp_shadow_sweep.csv"
    done = set()
    if os.path.isfile(path):
        old = pd.read_csv(path)
        for _, r in old.iterrows():
            tag = r["tag"] if "tag" in r else r["defense"]
            nsh = int(r["n_shadows_run"] if "n_shadows_run" in r else r.get("n_shadows", -1))
            done.add((r["dataset"], str(tag), nsh, int(r["seed"])))

    # Complete Chameleon n=64
    for tag, dn, dp in [
        ("none", "none", {}),
        ("lbp", "lbp", {"scale": 0.3}),
        ("harp", "harp", dict(LOCKED_HARP)),
    ]:
        for seed in SEEDS3:
            key = ("Chameleon", tag, 64, seed)
            if key in done:
                print("skip", key, flush=True)
                continue
            print(f"SHADOW Chameleon {tag} n=64 seed={seed}", flush=True)
            local = dict(c)
            local["lira"] = {"n_shadows": 64}
            t0 = time.time()
            r = run_one("Chameleon", "GraphSAGE", dn, dp, seed, device=device, config=local)
            r = dict(r)
            r["tag"] = tag
            r["n_shadows_run"] = 64
            r["wall_seconds"] = round(time.time() - t0, 2)
            append_csv(path, [r])
            print(
                f"  acc={r['test_accuracy']:.4f} lira={r['lira_attack_auc']:.4f} "
                f"tpr01={r.get('lira_tpr_at_0.01_fpr')} wall={r['wall_seconds']}",
                flush=True,
            )

    df = pd.read_csv(path)
    means = (
        df.groupby(["dataset", "tag", "n_shadows_run"])[
            ["test_accuracy", "lira_attack_auc", "lira_tpr_at_0.01_fpr", "lira_tpr_at_0.001_fpr"]
        ]
        .mean()
        .round(4)
        .reset_index()
    )
    means.to_csv(f"{OUT}/harp_shadow_sweep_means.csv", index=False)

    gaps = []
    for (ds, nsh), g in means.groupby(["dataset", "n_shadows_run"]):
        by = {row.tag: row for _, row in g.iterrows()}
        if not {"none", "lbp", "harp"} <= set(by):
            continue
        gaps.append(
            {
                "dataset": ds,
                "n_shadows": int(nsh),
                "none_lira": by["none"].lira_attack_auc,
                "lbp_lira": by["lbp"].lira_attack_auc,
                "harp_lira": by["harp"].lira_attack_auc,
                "harp_minus_none": round(by["harp"].lira_attack_auc - by["none"].lira_attack_auc, 4),
                "harp_minus_lbp": round(by["harp"].lira_attack_auc - by["lbp"].lira_attack_auc, 4),
                "none_tpr01": by["none"]["lira_tpr_at_0.01_fpr"],
                "lbp_tpr01": by["lbp"]["lira_tpr_at_0.01_fpr"],
                "harp_tpr01": by["harp"]["lira_tpr_at_0.01_fpr"],
                "harp_acc": by["harp"].test_accuracy,
                "lbp_acc": by["lbp"].test_accuracy,
                "delta_acc_vs_lbp": round(by["harp"].test_accuracy - by["lbp"].test_accuracy, 4),
            }
        )
    gap_df = pd.DataFrame(gaps)
    gap_df.to_csv(f"{OUT}/harp_shadow_comparative.csv", index=False)
    print(gap_df.to_string(index=False), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for ax, ds in zip(axes, ["Cora", "Chameleon"]):
        sub = means[means.dataset == ds]
        for tag, color, ls in [("none", "#444", "--"), ("lbp", "#c44e52", "-."), ("harp", "#4c72b0", "-")]:
            g = sub[sub.tag == tag].sort_values("n_shadows_run")
            if g.empty:
                continue
            ax.plot(g.n_shadows_run, g.lira_attack_auc, "o-", color=color, ls=ls, label=tag, lw=2)
        ax.set_title(ds)
        ax.set_xlabel(r"$n_{\mathrm{shadows}}$")
        ax.set_ylabel("LiRA AUROC")
        ax.set_ylim(0.45, 1.0)
        ax.axhline(0.5, color="gray", lw=0.7, alpha=0.6)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Shadow scaling: absolute LiRA rises for all defenses; Acc stays fixed", fontsize=10)
    fig.tight_layout()
    savefig(fig, "fig_harp_shadow_scaling.png")
    plt.close()
    return means


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------
def _train_pack(dataset, defense_name, defense_params, seed, device, c):
    split_kw = _split_kwargs(c)
    data, num_classes, num_features = _load_target_data(
        dataset, c["data_dir"], seed, bool(c.get("large_graph_use_official_split", True)), split_kw
    )
    tk = {
        "epochs": int(c.get("training", {}).get("epochs", 50)),
        "lr": float(c.get("training", {}).get("lr", 0.01)),
        "weight_decay": float(c.get("training", {}).get("weight_decay", 5e-4)),
        "device": device,
    }
    use_mb = dataset in MINIBATCH_DATASETS
    mb = c.get("minibatch", {})
    p, pr, risk, ts, dep, rel = _train_and_predict_gnn(
        "GraphSAGE",
        defense_name,
        defense_params,
        data,
        num_features,
        num_classes,
        device,
        tk["epochs"],
        tk["lr"],
        tk["weight_decay"],
        tk,
        None,
        use_mb,
        int(mb.get("batch_size", 1024)),
        mb.get("num_neighbors", [15, 10]),
        c,
        release_seed=seed,
        multi_query_k=1,
    )
    return data, np.asarray(p), rel


def run_failure_cases(device, c, seed: int = 42):
    data, p_none, _ = _train_pack("Chameleon", "none", {}, seed, device, c)
    _, p_harp, rel = _train_pack("Chameleon", "harp", dict(LOCKED_HARP), seed, device, c)
    scales, protected, seeds, stats = compute_harp_scales(
        data,
        k_hops=int(LOCKED_HARP["k_hops"]),
        strong_noise_scale=float(LOCKED_HARP["strong_noise_scale"]),
        weak_noise_scale=float(LOCKED_HARP["weak_noise_scale"]),
        use_lte=True,
        target_protect_frac=float(LOCKED_HARP["target_protect_frac"]),
        arch="sage",
    )
    yn = data.y.detach().cpu().numpy().reshape(-1)
    trm = data.train_mask.detach().cpu().numpy()
    ei = data.edge_index.detach().cpu().numpy()
    deg = np.bincount(np.concatenate([ei[0], ei[1]]), minlength=len(yn))

    conf_n = p_none[np.arange(len(yn)), yn]
    conf_h = p_harp[np.arange(len(yn)), yn]
    test_idx = np.flatnonzero(~trm)
    mean_tn, mean_th = float(conf_n[test_idx].mean()), float(conf_h[test_idx].mean())

    rows = []
    for v in np.flatnonzero(trm & protected):
        gap_n = conf_n[v] - mean_tn
        gap_h = conf_h[v] - mean_th
        rows.append(
            {
                "node": int(v),
                "label": int(yn[v]),
                "deg": int(deg[v]),
                "conf_none": float(conf_n[v]),
                "conf_harp": float(conf_h[v]),
                "gap_none": float(gap_n),
                "gap_harp": float(gap_h),
                "gap_reduced": float(gap_n - gap_h),
                "seed": bool(seeds[v]),
            }
        )
    df = pd.DataFrame(rows)
    fail = df[df.gap_reduced <= 0].sort_values("gap_harp", ascending=False)
    fail.head(10).to_csv(f"{OUT}/harp_failure_cases_chameleon.csv", index=False)

    # Characterize failure mode
    peaky = float((fail.conf_none > 0.9).mean()) if len(fail) else 0.0
    lowdeg = float((fail.deg <= np.median(df.deg)).mean()) if len(fail) else 0.0
    summary = {
        "n_protected_train": int(len(df)),
        "n_failure": int(len(fail)),
        "failure_rate": float(len(fail) / max(len(df), 1)),
        "mean_gap_none": float(df.gap_none.mean()),
        "mean_gap_harp": float(df.gap_harp.mean()),
        "frac_failures_conf_gt_0.9": peaky,
        "frac_failures_low_or_med_degree": lowdeg,
        "frac_protected": float(stats.get("frac_protected", protected.mean())),
        "narrative": (
            f"Of {len(df)} protected train nodes on Chameleon, "
            f"{len(fail)} ({100*len(fail)/max(len(df),1):.0f}%) do not shrink their "
            f"train−test confidence gap under HARP. Among failures, "
            f"{100*peaky:.0f}% already had clean true-class confidence >0.9: "
            "Laplace at σ=0.3 rarely erases an already near-one-hot cue after "
            "renormalization. Population LiRA therefore moves only modestly—"
            "the protector helps mid-confidence mass, not the overconfident tail."
        ),
    }
    with open(f"{OUT}/harp_failure_cases_chameleon.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(fail.head(10)[["node", "deg", "conf_none", "conf_harp", "gap_reduced"]].to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# ogbn-arxiv LiRA n=4
# ---------------------------------------------------------------------------
def run_ogbn_lira4(device):
    c = cfg("experiment_config_ogbn.yaml")
    c["lira"] = {"n_shadows": 4}
    c["large_graph_use_official_split"] = True
    path = f"{OUT}/harp_ogbn_lira4.csv"
    done = done_keys(path, ("tag", "seed"))
    harp = {**LOCKED_HARP, "use_gate": False, "warmup_epochs": 3}
    for tag, dn, dp in [
        ("none", "none", {}),
        ("lbp", "lbp", {"scale": 0.3}),
        ("harp", "harp", harp),
    ]:
        for seed in SEEDS3:
            if (tag, seed) in done:
                print("skip ogbn4", tag, seed, flush=True)
                continue
            print(f"OGBN-ARXIV LiRA4 {tag} seed={seed}", flush=True)
            t0 = time.time()
            r = run_one("ogbn-arxiv", "GraphSAGE", dn, dp, seed, device=device, config=c)
            r = dict(r)
            r["tag"] = tag
            r["n_shadows_run"] = 4
            r["wall_seconds"] = round(time.time() - t0, 2)
            append_csv(path, [r])
            print(
                f"  acc={r['test_accuracy']:.4f} lira={r['lira_attack_auc']:.4f} "
                f"wall={r['wall_seconds']}",
                flush=True,
            )
    if os.path.isfile(path):
        df = pd.read_csv(path)
        means = df.groupby("tag")[
            ["test_accuracy", "lira_attack_auc", "noise_mass", "frac_protected"]
        ].mean().round(4)
        means.to_csv(f"{OUT}/harp_ogbn_lira4_means.csv")
        print(means, flush=True)
        return df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Products subsample LiRA
# ---------------------------------------------------------------------------
def run_products_sub_lira(device, n_sub: int = 15000):
    path = f"{OUT}/harp_products_sub_lira.csv"
    sub_path = f"{OUT}/products_sub_{n_sub}.pt"
    c = cfg("experiment_config_ogbn.yaml")
    c["lira"] = {"n_shadows": 2}

    if not os.path.isfile(sub_path):
        print(f"Building products subsample n={n_sub}", flush=True)
        try:
            from torch_geometric.utils import subgraph
            from torch_geometric.data import Data
            from ogb_loader import load_large_benchmark

            # Existing processed data lives under data/ogb/
            root = os.path.join(c["data_dir"], "ogb")
            if not os.path.isdir(os.path.join(root, "ogbn_products", "processed")):
                root = c["data_dir"]
            data, num_classes, num_features = load_large_benchmark("ogbn-products", root)
            y = data.y.view(-1)
            n = data.num_nodes
            # BFS-grown induced subgraph (random sample is too sparse on products)
            from collections import deque

            ei_np = data.edge_index.detach().cpu().numpy()
            adj = [[] for _ in range(n)]
            for u, v in zip(ei_np[0], ei_np[1]):
                adj[int(u)].append(int(v))
            rng = np.random.RandomState(0)
            start = int(rng.randint(0, n))
            seen = {start}
            q = deque([start])
            while q and len(seen) < n_sub:
                u = q.popleft()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        q.append(v)
                        if len(seen) >= n_sub:
                            break
            keep = np.sort(np.fromiter(seen, dtype=np.int64))
            keep_t = torch.tensor(keep, dtype=torch.long)
            ei, _ = subgraph(keep_t, data.edge_index, relabel_nodes=True, num_nodes=n)
            sub = Data(x=data.x[keep_t].clone(), y=y[keep_t].clone(), edge_index=ei)
            torch.save({"data": sub, "keep": keep, "method": "bfs"}, sub_path)
            print(
                f"saved {sub_path} nodes={sub.num_nodes} edges={ei.size(1)} method=bfs",
                flush=True,
            )
            del data
        except Exception as e:
            print("products subsample build failed:", e, flush=True)
            traceback.print_exc()
            return pd.DataFrame()

    blob = torch.load(sub_path, weights_only=False)
    base = blob["data"]
    done = done_keys(path, ("tag", "seed"))
    harp = {**LOCKED_HARP, "use_gate": False, "warmup_epochs": 3}

    for tag, dn, dp in [
        ("none", "none", {}),
        ("lbp", "lbp", {"scale": 0.3}),
        ("harp", "harp", harp),
    ]:
        for seed in SEEDS3:
            if (tag, seed) in done:
                print("skip prod-sub", tag, seed, flush=True)
                continue
            print(f"PRODUCTS-SUB {tag} seed={seed}", flush=True)
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
            num_features = data.x.size(1)
            num_classes = int(data.y.max().item()) + 1
            tk = {
                "epochs": 30,
                "lr": 0.01,
                "weight_decay": 5e-4,
                "device": device,
            }
            # Target
            p, pr, _, _, _, rel = _train_and_predict_gnn(
                "GraphSAGE",
                dn,
                dp,
                data,
                num_features,
                num_classes,
                device,
                tk["epochs"],
                tk["lr"],
                tk["weight_decay"],
                tk,
                None,
                False,
                1024,
                [15, 10],
                c,
                release_seed=seed,
                multi_query_k=1,
            )
            yn = data.y.detach().cpu().numpy().reshape(-1)
            trm = data.train_mask.cpu().numpy()
            tem = data.test_mask.cpu().numpy()
            acc = float(accuracy_score(yn[tem], np.asarray(pr)[tem]))

            # 2 defense-aware shadows with reshuffled train halves
            sh_probs, sh_tr, sh_te = [], [], []
            train_idx = np.flatnonzero(trm)
            for s in range(2):
                rng_s = np.random.RandomState(seed * 17 + s)
                order = train_idx.copy()
                rng_s.shuffle(order)
                half = len(order) // 2
                for half_i, idx_half in enumerate([order[:half], order[half:]]):
                    sh = data.clone()
                    msk = torch.zeros(m, dtype=torch.bool)
                    msk[idx_half] = True
                    sh.train_mask = msk
                    sp, _, _, _, _, _ = _train_and_predict_gnn(
                        "GraphSAGE",
                        dn,
                        dp,
                        sh,
                        num_features,
                        num_classes,
                        device,
                        tk["epochs"],
                        tk["lr"],
                        tk["weight_decay"],
                        tk,
                        None,
                        False,
                        1024,
                        [15, 10],
                        c,
                        release_seed=seed * 100 + s * 2 + half_i,
                        multi_query_k=1,
                    )
                    sh_probs.append(np.asarray(sp))
                    sh_tr.append(msk.numpy())
                    sh_te.append(tem)

            auc, acc05, tpr001, tpr01 = lira_gaussian_auc(
                np.asarray(p), yn, trm, tem, sh_probs, sh_tr, sh_te
            )
            row = {
                "tag": tag,
                "seed": seed,
                "n_sub": m,
                "test_accuracy": acc,
                "lira_attack_auc": float(auc),
                "lira_tpr_at_0.01_fpr": float(tpr01),
                "noise_mass": rel.get("noise_mass", float("nan")),
                "frac_protected": rel.get("frac_protected", float("nan")),
                "wall_seconds": round(time.time() - t0, 2),
            }
            append_csv(path, [row])
            print(
                f"  acc={acc:.4f} lira={auc:.4f} tpr01={tpr01:.4f} wall={row['wall_seconds']}",
                flush=True,
            )

    if os.path.isfile(path):
        df = pd.read_csv(path)
        means = df.groupby("tag")[["test_accuracy", "lira_attack_auc", "lira_tpr_at_0.01_fpr"]].mean().round(4)
        means.to_csv(f"{OUT}/harp_products_sub_lira_means.csv")
        print(means, flush=True)
        return df
    return pd.DataFrame()


def main():
    device = torch.device("cpu")
    c = cfg()
    print("=== 1 local ε ===", flush=True)
    run_local_eps_accounting()
    print("=== 2 cache ===", flush=True)
    run_cache_simulation()
    print("=== 3 hop synth ===", flush=True)
    run_hop_leakage_synth()
    print("=== 4 shadow completion ===", flush=True)
    run_shadow_completion(device, c)
    print("=== 5 failure cases ===", flush=True)
    try:
        run_failure_cases(device, c)
    except Exception:
        traceback.print_exc()
    print("=== 6 ogbn-arxiv LiRA4 ===", flush=True)
    try:
        run_ogbn_lira4(device)
    except Exception:
        traceback.print_exc()
    print("=== 7 products sub LiRA ===", flush=True)
    try:
        run_products_sub_lira(device)
    except Exception:
        traceback.print_exc()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
