"""
HARP elevation experiments for IEEE BigData paper fixes.

1) Cora fairness + Frac sweep, 5 seeds + bootstrap CIs (incl. ECE)
2) Multi-query K under HARP / LBP / SAMI / equal-mass
3) Unprotected-slice conf + LiRA under HARP
4) Canary / top-decile conf + population LiRA for HARP vs baselines
5) Photo HARP via Amazon loader

Writes results/harp_*_5seed.csv, harp_multi_query.csv, harp_unprot_lira.csv,
harp_canary.csv, harp_photo.csv, harp_elevation_summary.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Amazon

from config import ensure_dirs, load_config
from data import load_citation, resplit
from defenses.harp import LOCKED_HARP, compute_harp_scales
from defenses.sami import compute_lte_risk
from experiment import (
    _make_shadow_data,
    _split_kwargs,
    _train_and_predict_gnn,
    run_one,
)
from lira_attack import lira_auc_on_subset, lira_gaussian_scores
from stats_utils import bootstrap_ci_over_seeds, bootstrap_delta_ci

SEEDS5 = [42, 123, 456, 789, 1024]
SEEDS3 = [42, 123, 456]
LOCKED_SAMI = {
    "lam": 0.5,
    "use_lte": True,
    "use_gate": True,
    "arch_aware": True,
    "noise_scale": 0.35,
    "budget_B": 0.0,
    "warmup_epochs": 5,
    "entropy_coef": 0.05,
}


def _cfg():
    cfg = dict(load_config("experiment_config_confirmatory.yaml"))
    ensure_dirs(cfg)
    cfg["lira"] = {"n_shadows": 4}
    cfg["attacks"] = ["confidence", "lira"]
    return cfg


def run_fairness_and_frac(device, cfg):
    rows = []
    for seed in SEEDS5:
        for name, dn, dp in [
            ("none", "none", {}),
            ("lbp_strong", "lbp", {"scale": 0.3}),
            ("lbp_equal_mass", "lbp", {"scale": 0.120}),
            ("harp", "harp", dict(LOCKED_HARP)),
            ("sami", "sami", dict(LOCKED_SAMI)),
        ]:
            print(f"EQ5 {name} seed={seed}", flush=True)
            r = run_one("Cora", "GraphSAGE", dn, dp, seed, device=device, config=cfg)
            r["tag"] = name
            rows.append(r)
            pd.DataFrame(rows).to_csv("results/harp_fairness_cora_5seed.csv", index=False)
    eq = pd.DataFrame(rows)

    rows2 = []
    for frac in [0.20, 0.30, 0.40, 0.60, 1.0]:
        for seed in SEEDS5:
            if frac >= 1.0:
                dp = {
                    **LOCKED_HARP,
                    "risk_frac": 1.0,
                    "k_hops": 0,
                    "target_protect_frac": None,
                    "use_gate": False,
                    "train_on_protected": False,
                }
                dn = "harp_uniform"
            else:
                dp = {**LOCKED_HARP, "target_protect_frac": frac}
                dn = "harp"
            print(f"FRAC5 {frac} seed={seed}", flush=True)
            r = run_one("Cora", "GraphSAGE", dn, dp, seed, device=device, config=cfg)
            r["target_frac"] = frac
            r["tag"] = f"harp_frac{frac}"
            rows2.append(r)
            pd.DataFrame(rows2).to_csv("results/harp_frac_sweep_5seed.csv", index=False)
    fs = pd.DataFrame(rows2)

    eq2 = eq.copy()
    eq2["defense"] = eq2["tag"]
    eq2["dataset"] = "Cora"
    eq2["model"] = "GraphSAGE"
    boot = bootstrap_ci_over_seeds(
        eq2,
        group_cols=("dataset", "model", "defense"),
        metric_cols=[
            "test_accuracy",
            "lira_attack_auc",
            "ece_test",
            "noise_mass",
            "frac_protected",
        ],
    )
    boot.to_csv("results/harp_fairness_bootstrap.csv", index=False)

    deltas = []
    for defense in ["harp", "lbp_equal_mass", "sami"]:
        for value_col in ["test_accuracy", "lira_attack_auc", "ece_test"]:
            d = bootstrap_delta_ci(
                eq2,
                value_col=value_col,
                baseline="lbp_strong",
                defense=defense,
                group_cols=("dataset", "model"),
                n_resamples=2000,
            )
            if len(d):
                d["value_col"] = value_col
                d["baseline_tag"] = "lbp_strong"
                deltas.append(d)
    for defense in ["harp", "lbp_equal_mass", "lbp_strong", "sami"]:
        d = bootstrap_delta_ci(
            eq2,
            value_col="test_accuracy",
            baseline="none",
            defense=defense,
            group_cols=("dataset", "model"),
            n_resamples=2000,
        )
        if len(d):
            d["value_col"] = "test_accuracy"
            d["baseline_tag"] = "none"
            deltas.append(d)
    if deltas:
        pd.concat(deltas, ignore_index=True).to_csv(
            "results/harp_fairness_delta_bootstrap.csv", index=False
        )

    # Frac bootstrap
    fs2 = fs.copy()
    fs2["defense"] = fs2["tag"]
    fs2["dataset"] = "Cora"
    fs2["model"] = "GraphSAGE"
    boot_f = bootstrap_ci_over_seeds(
        fs2,
        group_cols=("dataset", "model", "defense"),
        metric_cols=["test_accuracy", "lira_attack_auc", "noise_mass"],
    )
    boot_f.to_csv("results/harp_frac_bootstrap.csv", index=False)

    print("Fairness means (5 seeds):\n", eq.groupby("tag")[
        ["test_accuracy", "lira_attack_auc", "ece_test", "noise_mass"]
    ].mean())
    print("Frac means (5 seeds):\n", fs.groupby("target_frac")[
        ["test_accuracy", "lira_attack_auc", "noise_mass"]
    ].mean())
    return eq, fs, boot


def run_multi_query(device, cfg):
    rows = []
    for k in [1, 5, 20]:
        local = dict(cfg)
        local["multi_query_k"] = k
        for name, dn, dp in [
            ("none", "none", {}),
            ("lbp_strong", "lbp", {"scale": 0.3}),
            ("lbp_equal_mass", "lbp", {"scale": 0.120}),
            ("harp", "harp", dict(LOCKED_HARP)),
            ("sami", "sami", dict(LOCKED_SAMI)),
        ]:
            for seed in SEEDS3:
                print(f"MQ K={k} {name} seed={seed}", flush=True)
                r = run_one("Cora", "GraphSAGE", dn, dp, seed, device=device, config=local)
                r["tag"] = name
                r["multi_query_k"] = k
                rows.append(r)
                pd.DataFrame(rows).to_csv("results/harp_multi_query.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["tag", "multi_query_k"])[
        ["test_accuracy", "lira_attack_auc", "conf_attack_auc"]
    ].mean())
    return df


def run_unprotected_lira(device, cfg):
    """Population vs unprotected-slice conf + LiRA under locked HARP."""
    split_kw = _split_kwargs(cfg)
    ep = int(cfg.get("training", {}).get("epochs", 50))
    lr = float(cfg.get("training", {}).get("lr", 0.01))
    wd = float(cfg.get("training", {}).get("weight_decay", 5e-4))
    rows = []
    for seed in SEEDS5:
        print(f"UNPROT LiRA seed={seed}", flush=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        from experiment import _load_target_data
        data, nc, nf = _load_target_data("Cora", cfg["data_dir"], seed, True, split_kw)

        p, pr, risk, train_s, _, stats = _train_and_predict_gnn(
            "GraphSAGE", "harp", dict(LOCKED_HARP), data, nf, nc, device,
            ep, lr, wd, {}, None, False, 1024, [15, 10], cfg,
            release_seed=seed, multi_query_k=1,
        )
        # Recompute protected mask (same LTE risk)
        scales, prot, _, hstats = compute_harp_scales(
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

        conf = p[np.arange(len(yn)), yn]

        def _conf_auc(m_mask, n_mask):
            if m_mask.sum() < 5 or n_mask.sum() < 5:
                return float("nan")
            s = np.concatenate([conf[m_mask], conf[n_mask]])
            y = np.concatenate([np.ones(int(m_mask.sum())), np.zeros(int(n_mask.sum()))])
            return float(roc_auc_score(y, s))

        conf_pop = _conf_auc(trm, tem)
        conf_un = _conf_auc(trm & unprot, tem & unprot)
        conf_pr = _conf_auc(trm & prot, tem & prot)

        # Shadows
        shadow_probs, shadow_tr, shadow_te = [], [], []
        for k in range(4):
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
        lira_pop, _, _, _ = __import__("lira_attack", fromlist=["lira_gaussian_auc"]).lira_gaussian_auc(
            p, yn, trm, tem, shadow_probs, shadow_tr, shadow_te
        )
        lira_un, n_um, n_un = lira_auc_on_subset(
            scores, y_mem, np.where(trm & unprot)[0], np.where(tem & unprot)[0]
        )
        lira_pr, n_pm, n_pn = lira_auc_on_subset(
            scores, y_mem, np.where(trm & prot)[0], np.where(tem & prot)[0]
        )
        # Acc
        acc = float((pr[tem] == yn[tem]).mean())
        rows.append({
            "seed": seed,
            "acc": acc,
            "frac_prot": float(hstats["frac_protected"]),
            "conf_pop": conf_pop,
            "conf_unprot": conf_un,
            "conf_prot": conf_pr,
            "lira_pop": float(lira_pop),
            "lira_unprot": float(lira_un),
            "lira_prot": float(lira_pr),
            "n_unprot_train": int(n_um),
            "n_unprot_test": int(n_un),
            "n_prot_train": int(n_pm),
            "n_prot_test": int(n_pn),
            "train_seconds": float(train_s),
        })
        pd.DataFrame(rows).to_csv("results/harp_unprot_lira.csv", index=False)
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print("UNPROT means:\n", df.mean(numeric_only=True))
    return df


def run_canary(device, cfg):
    """Top-decile LTE conf + planted-canary conf + population LiRA."""
    from run_canary_lira import _plant_canaries

    rows = []
    for defense, dn, dp in [
        ("none", "none", {}),
        ("lbp", "lbp", {"scale": 0.3}),
        ("sami", "sami", dict(LOCKED_SAMI)),
        ("harp", "harp", dict(LOCKED_HARP)),
    ]:
        for seed in SEEDS3:
            print(f"CANARY {defense} seed={seed}", flush=True)
            rfull = run_one("Cora", "GraphSAGE", dn, dp, seed, device=device, config=cfg)

            # Top-decile conf on released scores: re-run predict path
            split_kw = _split_kwargs(cfg)
            data, nc, nf = __import__("experiment", fromlist=["_load_target_data"])._load_target_data(
                "Cora", cfg["data_dir"], seed, True, split_kw
            )
            ep = int(cfg.get("training", {}).get("epochs", 50))
            lr = float(cfg.get("training", {}).get("lr", 0.01))
            wd = float(cfg.get("training", {}).get("weight_decay", 5e-4))
            p, pr, risk, _, _, _ = _train_and_predict_gnn(
                "GraphSAGE", dn, dp, data, nf, nc, device,
                ep, lr, wd, {}, None, False, 1024, [15, 10], cfg,
                release_seed=seed, multi_query_k=1,
            )
            yn = data.y.numpy()
            trm = data.train_mask.numpy()
            tem = data.test_mask.numpy()
            rnp = risk.numpy() if risk is not None else compute_lte_risk(data).numpy()
            thr = np.quantile(rnp[trm], 0.9)
            top = trm & (rnp >= thr)
            conf = p[np.arange(len(yn)), yn]

            def _auc(mem, non):
                if mem.sum() < 5 or non.sum() < 5:
                    return float("nan")
                s = np.concatenate([conf[mem], conf[non]])
                y = np.concatenate([np.ones(int(mem.sum())), np.zeros(int(non.sum()))])
                return float(roc_auc_score(y, s))

            top_conf = _auc(top, tem)

            # Planted canaries
            dcan, can_idx = _plant_canaries(data, seed, k=64)
            can_idx = np.asarray(can_idx, dtype=int)
            p_c, _, risk_c, _, _, _ = _train_and_predict_gnn(
                "GraphSAGE", dn, dp, dcan, nf, nc, device,
                ep, lr, wd, {}, None, False, 1024, [15, 10], cfg,
                release_seed=seed, multi_query_k=1,
            )
            yc = dcan.y.numpy()
            conf_c = p_c[np.arange(len(yc)), yc]
            te2 = dcan.test_mask.numpy()
            if len(can_idx) > 5 and te2.sum() > 5:
                s = np.concatenate([conf_c[can_idx], conf_c[te2]])
                y = np.concatenate([np.ones(len(can_idx)), np.zeros(int(te2.sum()))])
                can_conf = float(roc_auc_score(y, s))
            else:
                can_conf = float("nan")

            rows.append({
                "defense": defense,
                "seed": seed,
                "acc": float(rfull["test_accuracy"]),
                "lira_pop": float(rfull["lira_attack_auc"]),
                "conf_pop": float(rfull["conf_attack_auc"]),
                "top_decile_conf_auc": top_conf,
                "planted_canary_conf_auc": can_conf,
                "ece_test": float(rfull.get("ece_test", np.nan)),
            })
            pd.DataFrame(rows).to_csv("results/harp_canary.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby("defense").mean(numeric_only=True))
    return df


def run_photo(device, cfg):
    import experiment as exp

    def load_photo(name, data_dir=None):
        d = Amazon(root=os.path.join(data_dir or cfg["data_dir"], "Photo"), name="Photo")
        return d[0], d.num_classes, d.num_features

    _orig_t = exp._load_target_data
    _orig_s = exp._make_shadow_data

    def _load_t(dataset_name, data_dir, seed, use_official_large, split_kw):
        if dataset_name == "Photo":
            data, nc, nf = load_photo(dataset_name, data_dir)
            ratios = exp._resplit_kwargs(split_kw)
            return resplit(data, seed, **ratios), nc, nf
        return _orig_t(dataset_name, data_dir, seed, use_official_large, split_kw)

    def _load_s(dataset_name, data_dir, shadow_seed, split_kw):
        if dataset_name == "Photo":
            data, nc, nf = load_photo(dataset_name, data_dir)
            ratios = exp._resplit_kwargs(split_kw)
            return resplit(data, shadow_seed, **ratios), nc, nf
        return _orig_s(dataset_name, data_dir, shadow_seed, split_kw)

    exp._load_target_data = _load_t
    exp._make_shadow_data = _load_s
    rows = []
    try:
        for dn, dp in [
            ("none", {}),
            ("lbp", {"scale": 0.3}),
            ("harp", dict(LOCKED_HARP)),
            ("sami", dict(LOCKED_SAMI)),
        ]:
            for seed in SEEDS3:
                print(f"Photo/{dn} seed={seed}", flush=True)
                rows.append(run_one("Photo", "GraphSAGE", dn, dp, seed, device=device, config=cfg))
                pd.DataFrame(rows).to_csv("results/harp_photo.csv", index=False)
    finally:
        exp._load_target_data = _orig_t
        exp._make_shadow_data = _orig_s
    df = pd.DataFrame(rows)
    print(df.groupby("defense")[
        ["test_accuracy", "lira_attack_auc", "noise_mass", "frac_protected"]
    ].mean())
    return df


def main():
    t0 = time.time()
    device = torch.device("cpu")
    cfg = _cfg()
    os.makedirs("results", exist_ok=True)

    print("=== 1) Fairness + Frac (5 seeds) ===", flush=True)
    eq, fs, boot = run_fairness_and_frac(device, cfg)

    print("=== 2) Multi-query ===", flush=True)
    mq = run_multi_query(device, cfg)

    print("=== 3) Unprotected LiRA ===", flush=True)
    un = run_unprotected_lira(device, cfg)

    print("=== 4) Canaries ===", flush=True)
    can = run_canary(device, cfg)

    print("=== 5) Photo ===", flush=True)
    photo = run_photo(device, cfg)

    summary = {
        "wall_s": round(time.time() - t0, 1),
        "fairness_means": eq.groupby("tag")[
            ["test_accuracy", "lira_attack_auc", "ece_test", "noise_mass"]
        ].mean().round(4).to_dict(),
        "bootstrap_acc_harp": boot[(boot.defense == "harp") & (boot.metric == "test_accuracy")]
        .round(4).to_dict(orient="records"),
        "bootstrap_ece": boot[boot.metric == "ece_test"].round(4).to_dict(orient="records"),
        "frac_means": fs.groupby("target_frac")[
            ["test_accuracy", "lira_attack_auc", "noise_mass"]
        ].mean().round(4).to_dict(),
        "multi_query": mq.groupby(["tag", "multi_query_k"])[
            ["lira_attack_auc", "conf_attack_auc"]
        ].mean().round(4).to_dict(),
        "unprot": un.mean(numeric_only=True).round(4).to_dict(),
        "canary": can.groupby("defense").mean(numeric_only=True).round(4).to_dict(),
        "photo": photo.groupby("defense")[
            ["test_accuracy", "lira_attack_auc", "noise_mass"]
        ].mean().round(4).to_dict(),
    }
    with open("results/harp_elevation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("DONE", summary["wall_s"], "s", flush=True)


if __name__ == "__main__":
    main()
