"""
Submission-polish experiments for HARP BigData paper:
1) Equal-mass LBP (b=0.12) on Citeseer/Chameleon/Actor (+ECE)
2) Slice ECE + Brier on Cora under locked HARP (clean vs protected)
3) Session-budget policy: fresh-draw cap B vs uncapped multi-query K
4) Localized clean-slice confidence (already in unprot; refresh summary JSON)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from attacks import average_posterior_queries, calibration_error
from config import ensure_dirs, load_config
from defenses.harp import LOCKED_HARP, compute_harp_scales
from experiment import _load_target_data, _train_and_predict_gnn, run_one
from stats_utils import bootstrap_delta_ci

SEEDS3 = [42, 123, 456]
SEEDS5 = [42, 123, 456, 789, 1024]
EQ_SCALE = 0.12  # Mass_HARP / n = 0.4 * 0.30


def _brier(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    onehot = np.zeros((len(y), n_classes), dtype=float)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def run_equal_mass(device, cfg) -> pd.DataFrame:
    datasets = ["Citeseer", "Chameleon", "Actor"]
    rows: List[Dict] = []
    out = "results/harp_equal_mass_multids.csv"
    for ds in datasets:
        for seed in SEEDS3:
            print(f"EQ-MASS {ds} seed={seed}", flush=True)
            for tag, dn, dp in [
                ("none", "none", {}),
                ("lbp_strong", "lbp", {"scale": 0.3}),
                ("lbp_equal_mass", "lbp", {"scale": EQ_SCALE}),
                ("harp", "harp", dict(LOCKED_HARP)),
            ]:
                r = run_one(ds, "GraphSAGE", dn, dp, seed, device=device, config=cfg)
                r = dict(r)
                r["tag"] = tag
                rows.append(r)
                pd.DataFrame(rows).to_csv(out, index=False)
                print(
                    f"  {tag}: acc={r['test_accuracy']:.4f} lira={r['lira_attack_auc']:.4f} "
                    f"ece={r.get('ece_test')} mass={r.get('noise_mass')}",
                    flush=True,
                )
    df = pd.DataFrame(rows)
    means = (
        df.groupby(["dataset", "tag"])[["test_accuracy", "lira_attack_auc", "ece_test", "noise_mass"]]
        .mean()
        .round(4)
    )
    print(means, flush=True)
    means.to_csv("results/harp_equal_mass_multids_means.csv")
    # paired ΔECE HARP - equal-mass
    deltas = []
    for ds in datasets:
        sub = df[df.dataset == ds].copy()
        sub["defense"] = sub["tag"]
        # bootstrap_delta_ci expects (dataset, model) groups
        sub = sub.copy()
        sub["model"] = "GraphSAGE"
        d = bootstrap_delta_ci(
            sub,
            value_col="ece_test",
            baseline="lbp_equal_mass",
            defense="harp",
            group_cols=("dataset", "model"),
            n_resamples=2000,
        )
        if len(d):
            deltas.append(d)
    if deltas:
        pd.concat(deltas).to_csv("results/harp_equal_mass_multids_delta_ece.csv", index=False)
    return df


def run_slice_ece(device, cfg) -> pd.DataFrame:
    """Population + clean-slice + protected-slice ECE/Brier under locked HARP (no shadows)."""
    split_kw = {
        "train_ratio": float(cfg.get("split", {}).get("train_ratio", 0.4)),
        "val_ratio": float(cfg.get("split", {}).get("val_ratio", 0.2)),
        "test_ratio": float(cfg.get("split", {}).get("test_ratio", 0.4)),
    }
    ep = int(cfg.get("training", {}).get("epochs", 50))
    lr = float(cfg.get("training", {}).get("lr", 0.01))
    wd = float(cfg.get("training", {}).get("weight_decay", 5e-4))
    rows = []
    out = "results/harp_slice_ece.csv"
    for seed in SEEDS5:
        print(f"SLICE-ECE seed={seed}", flush=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        data, nc, nf = _load_target_data("Cora", cfg["data_dir"], seed, True, split_kw)
        p, pr, risk, train_s, _, stats = _train_and_predict_gnn(
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
        # Also release-only for calibration attribution
        p_ro, pr_ro, risk_ro, _, _, _ = _train_and_predict_gnn(
            "GraphSAGE",
            "harp",
            {**LOCKED_HARP, "lam": 0.0, "train_on_protected": False},
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
        yn = data.y.numpy()
        tem = data.test_mask.numpy()
        un = ~prot

        def pack(tag, probs, preds):
            return {
                "seed": seed,
                "variant": tag,
                "acc": float((preds[tem] == yn[tem]).mean()),
                "ece_pop": calibration_error(probs[tem], yn[tem]),
                "ece_clean": calibration_error(probs[tem & un], yn[tem & un]),
                "ece_prot": calibration_error(probs[tem & prot], yn[tem & prot]),
                "brier_pop": _brier(probs[tem], yn[tem], nc),
                "brier_clean": _brier(probs[tem & un], yn[tem & un], nc),
                "brier_prot": _brier(probs[tem & prot], yn[tem & prot], nc),
                "frac_prot": float(hstats["frac_protected"]),
                "n_clean_test": int((tem & un).sum()),
                "n_prot_test": int((tem & prot).sum()),
                "train_seconds": float(train_s),
            }

        rows.append(pack("harp", p, pr))
        rows.append(pack("harp_release_only", p_ro, pr_ro))
        # undefended reference on same split
        p0, pr0, _, _, _, _ = _train_and_predict_gnn(
            "GraphSAGE",
            "none",
            {},
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
        rows.append(
            {
                "seed": seed,
                "variant": "none",
                "acc": float((pr0[tem] == yn[tem]).mean()),
                "ece_pop": calibration_error(p0[tem], yn[tem]),
                "ece_clean": calibration_error(p0[tem], yn[tem]),
                "ece_prot": float("nan"),
                "brier_pop": _brier(p0[tem], yn[tem], nc),
                "brier_clean": _brier(p0[tem], yn[tem], nc),
                "brier_prot": float("nan"),
                "frac_prot": 0.0,
                "n_clean_test": int(tem.sum()),
                "n_prot_test": 0,
                "train_seconds": float("nan"),
            }
        )
        pd.DataFrame(rows).to_csv(out, index=False)
        print(rows[-3], flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby("variant")[["ece_pop", "ece_clean", "ece_prot", "brier_pop"]].mean().round(4))
    return df


def run_session_budget(device, cfg, dataset: str = "Cora") -> pd.DataFrame:
    """
    Systems session policy: at most B fresh Laplace draws per node; further draws
    reuse the last sample (no extra denoising). Compare to uncapped averaging.
    Also evaluate compounding: query t uses scale σ * max(1, t/B).
    """
    from defenses.sami import risk_scaled_posterior_noise
    from lira_attack import lira_gaussian_auc
    from experiment import _make_shadow_data

    split_kw = {
        "train_ratio": float(cfg.get("split", {}).get("train_ratio", 0.4)),
        "val_ratio": float(cfg.get("split", {}).get("val_ratio", 0.2)),
        "test_ratio": float(cfg.get("split", {}).get("test_ratio", 0.4)),
    }
    ep = int(cfg.get("training", {}).get("epochs", 50))
    lr = float(cfg.get("training", {}).get("lr", 0.01))
    wd = float(cfg.get("training", {}).get("weight_decay", 5e-4))
    rows = []
    out = (
        "results/harp_session_budget.csv"
        if dataset == "Cora"
        else f"results/harp_session_budget_{dataset.lower()}.csv"
    )
    B = 5
    K = 20

    for seed in SEEDS3:
        print(f"SESSION {dataset} seed={seed}", flush=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        data, nc, nf = _load_target_data(dataset, cfg["data_dir"], seed, True, split_kw)
        # Get clean base posteriors + HARP scales via one train
        # Train with HARP alignment, then manually release under policies
        p_base, pr, risk, train_s, _, stats = _train_and_predict_gnn(
            "GraphSAGE",
            "harp",
            {**LOCKED_HARP, "strong_noise_scale": 0.0, "weak_noise_scale": 0.0},  # clean scores after train
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
        scales, prot, _, hstats = compute_harp_scales(
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
        risk_arr = np.asarray(risk, dtype=float).reshape(-1)
        # Use binary risk for HARP release: scale vector as risk*strong for sami helper
        # risk_scaled uses risk * scale; set risk=1 on protected, 0 else, scale=strong
        risk_bin = (scales > 1e-12).astype(float)
        strong = float(LOCKED_HARP["strong_noise_scale"])

        def release_policy(policy: str, k: int, budget: int) -> np.ndarray:
            if policy == "uncapped_avg":
                return average_posterior_queries(p_base, risk_bin, strong, k, seed0=seed)
            if policy == "budget_reuse":
                # Only `budget` fresh draws; extras reuse last draw (avg of <=budget unique)
                bb = min(k, budget)
                return average_posterior_queries(p_base, risk_bin, strong, bb, seed0=seed)
            if policy == "compounding":
                # Query t uses scale strong * max(1, ceil(t/budget))
                acc = np.zeros_like(p_base, dtype=float)
                for t in range(1, k + 1):
                    mult = max(1.0, float(np.ceil(t / float(budget))))
                    acc += risk_scaled_posterior_noise(
                        p_base, risk_bin, scale=strong * mult, seed=int(seed) + 17 * t + 1
                    )
                return acc / float(k)
            raise ValueError(policy)

        # Shadows under uncapped K=1 for LiRA (defense-aware single-draw)
        shadow_probs, shadow_tr, shadow_te = [], [], []
        for j in range(4):
            ss = int(seed + 999 + j * 10007)
            np.random.seed(ss)
            torch.manual_seed(ss)
            sdata, _, _ = _make_shadow_data(dataset, cfg["data_dir"], ss, split_kw)
            sp, _, srisk, _, _, _ = _train_and_predict_gnn(
                "GraphSAGE",
                "harp",
                {**LOCKED_HARP, "strong_noise_scale": 0.0, "weak_noise_scale": 0.0},
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
                release_seed=ss,
                multi_query_k=1,
            )
            sscales, _, _, _ = compute_harp_scales(
                sdata.cpu(),
                risk=srisk,
                risk_frac=LOCKED_HARP["risk_frac"],
                k_hops=LOCKED_HARP["k_hops"],
                strong_noise_scale=strong,
                weak_noise_scale=0.0,
                target_protect_frac=LOCKED_HARP["target_protect_frac"],
                arch="sage",
                arch_aware=True,
            )
            srisk_bin = (np.asarray(sscales) > 1e-12).astype(float)
            # shadows always single-draw (attacker trains on one-shot API)
            sp_rel = average_posterior_queries(sp, srisk_bin, strong, 1, seed0=ss)
            shadow_probs.append(sp_rel)
            shadow_tr.append(sdata.train_mask.numpy())
            shadow_te.append(sdata.test_mask.numpy())

        for policy in ["uncapped_avg", "budget_reuse", "compounding"]:
            for k in [1, 5, 20]:
                pref = release_policy(policy, k, B)
                pred = pref.argmax(1)
                acc = float((pred[tem] == yn[tem]).mean())
                ece = calibration_error(pref[tem], yn[tem])
                lira, _, _, _ = lira_gaussian_auc(
                    pref, yn, trm, tem, shadow_probs, shadow_tr, shadow_te
                )
                # clean-slice conf AUROC (localized/adaptive client)
                prot = scales > 1e-12
                un = ~prot
                conf = pref[np.arange(len(yn)), yn]

                def _auc(mm, nm):
                    if mm.sum() < 5 or nm.sum() < 5:
                        return float("nan")
                    s = np.concatenate([conf[mm], conf[nm]])
                    y = np.concatenate([np.ones(int(mm.sum())), np.zeros(int(nm.sum()))])
                    return float(roc_auc_score(y, s))

                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "policy": policy,
                    "K": k,
                    "B": B,
                    "acc": acc,
                    "lira": float(lira),
                    "ece": ece,
                    "conf_unprot": _auc(trm & un, tem & un),
                    "conf_pop": _auc(trm, tem),
                    "train_seconds": float(train_s),
                }
                rows.append(row)
                print(row, flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["policy", "K"])[["acc", "lira", "ece", "conf_unprot"]].mean().round(4))
    return df


def main():
    os.environ.setdefault("PRIVACYGNN_CONFIG", "experiment_config_confirmatory.yaml")
    cfg = dict(load_config("experiment_config_confirmatory.yaml"))
    ensure_dirs(cfg)
    cfg["lira"] = {"n_shadows": 4}
    cfg["attacks"] = ["confidence", "lira"]
    device = torch.device("cpu")

    print("=== SLICE ECE ===", flush=True)
    slice_df = run_slice_ece(device, cfg)

    print("=== SESSION BUDGET ===", flush=True)
    sess_df = run_session_budget(device, cfg)

    print("=== EQUAL-MASS MULTI-DS ===", flush=True)
    eq_df = run_equal_mass(device, cfg)

    summary = {
        "slice_ece_means": slice_df.groupby("variant")[["ece_pop", "ece_clean", "ece_prot", "brier_pop"]]
        .mean()
        .round(4)
        .to_dict(),
        "session_means": sess_df.groupby(["policy", "K"])[["acc", "lira", "ece"]]
        .mean()
        .round(4)
        .to_dict(),
        "equal_mass_note": "b=0.12 matches Mass_HARP/n under Frac=0.40, sigma=0.30",
    }
    with open("results/harp_submission_polish_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
