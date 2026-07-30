"""
Cora GraphSAGE aggressive SAMI sweep (val-selected Acc+privacy proxy).

Grid: σ ∈ {0.35,0.45,0.55,0.65} × λ ∈ {0.5,0.75,1.0}
Selection: maximize val Acc subject to val conf-AUROC (train vs val) drop;
never uses test LiRA for selection. Report locked vs aggressive on test LiRA.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

from attacks import extract_features
from config import ensure_dirs, load_config
from data import load_citation, resplit
from defenses.sami import compute_lte_risk, train_gnn_sami, risk_scaled_posterior_noise
from experiment import _resplit_kwargs, _split_kwargs, _make_shadow_data
from lira_attack import lira_gaussian_auc
from models import SAGE

SIGMAS = [0.35, 0.45, 0.55, 0.65]
LAMS = [0.5, 0.75, 1.0]
SELECT_SEEDS = [42, 123]  # val selection
CONFIRM_SEEDS = [42, 123, 456, 789, 1024]
N_SHADOWS = 4


def _conf_auroc(p, y, a, b):
    fa = extract_features(p[a], y[a])[:, 1]
    fb = extract_features(p[b], y[b])[:, 1]
    scores = np.concatenate([fa, fb])
    lab = np.concatenate([np.ones(len(fa)), np.zeros(len(fb))])
    try:
        return float(roc_auc_score(lab, scores))
    except Exception:
        return float("nan")


def _train_eval(data, nf, nc, device, lam, sigma, seed):
    model = SAGE(nf, 64, nc)
    train_gnn_sami(
        model, data, device, epochs=50,
        lam=lam, use_lte=True, use_gate=True, arch_aware=True,
        warmup_epochs=5, entropy_coef=0.05,
    )
    risk = compute_lte_risk(data, arch="sage", arch_aware=True)
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        p = F.softmax(logits.cpu(), 1).numpy()
    p = risk_scaled_posterior_noise(p, risk.numpy(), scale=sigma, seed=seed)
    y = data.y.numpy()
    tr, va, te = data.train_mask.numpy(), data.val_mask.numpy(), data.test_mask.numpy()
    return {
        "acc_val": float(accuracy_score(y[va], p[va].argmax(1))),
        "acc_test": float(accuracy_score(y[te], p[te].argmax(1))),
        "conf_tr_va": _conf_auroc(p, y, tr, va),
        "conf_tr_te": _conf_auroc(p, y, tr, te),
        "p": p,
        "risk": risk,
        "model": model,
        "y": y, "tr": tr, "te": te,
    }


def _lira(dataset, cfg, split_kw, defense_params, seed, p_target, data, nf, nc, device):
    # retrain shadows with same lam; release with sigma
    lam = defense_params["lam"]
    sigma = defense_params["noise_scale"]
    shadow_p, shadow_tr, shadow_te = [], [], []
    for k in range(N_SHADOWS):
        sh_seed = seed + 999 + k * 10007
        sh, _, _ = _make_shadow_data(dataset, cfg["data_dir"], sh_seed, split_kw)
        m = SAGE(nf, 64, nc)
        train_gnn_sami(
            m, sh, device, epochs=50,
            lam=lam, use_lte=True, use_gate=True, arch_aware=True,
            warmup_epochs=5, entropy_coef=0.05,
        )
        risk = compute_lte_risk(sh, arch="sage", arch_aware=True)
        m.eval()
        with torch.no_grad():
            logits = m(sh.x.to(device), sh.edge_index.to(device))
            sp = F.softmax(logits.cpu(), 1).numpy()
        sp = risk_scaled_posterior_noise(sp, risk.numpy(), scale=sigma, seed=sh_seed)
        shadow_p.append(sp)
        shadow_tr.append(sh.train_mask.numpy())
        shadow_te.append(sh.test_mask.numpy())
    y = data.y.numpy()
    tr, te = data.train_mask.numpy(), data.test_mask.numpy()
    auc, _, _, tpr = lira_gaussian_auc(p_target, y, tr, te, shadow_p, shadow_tr, shadow_te)
    return float(auc), float(tpr)


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    device = torch.device(cfg.get("device", "cpu"))
    split_kw = _split_kwargs(cfg)
    ratios = _resplit_kwargs(split_kw)
    data0, nc, nf = load_citation("Cora", cfg["data_dir"])

    # --- selection grid (val only) ---
    rows = []
    for lam in LAMS:
        for sigma in SIGMAS:
            for seed in SELECT_SEEDS:
                data = resplit(data0.clone(), seed, **ratios)
                # baseline none conf for proxy delta
                from training import train_gnn
                m0 = SAGE(nf, 64, nc)
                train_gnn(m0, data, device, epochs=50)
                m0.eval()
                with torch.no_grad():
                    p0 = F.softmax(m0(data.x.to(device), data.edge_index.to(device)).cpu(), 1).numpy()
                y = data.y.numpy()
                tr, va = data.train_mask.numpy(), data.val_mask.numpy()
                none_conf = _conf_auroc(p0, y, tr, va)
                none_acc = float(accuracy_score(y[va], p0[va].argmax(1)))

                ev = _train_eval(data, nf, nc, device, lam, sigma, seed)
                rows.append({
                    "lam": lam, "sigma": sigma, "seed": seed,
                    "acc_val": ev["acc_val"], "acc_test": ev["acc_test"],
                    "conf_tr_va": ev["conf_tr_va"],
                    "none_conf_tr_va": none_conf,
                    "none_acc_val": none_acc,
                    "delta_conf_va": ev["conf_tr_va"] - none_conf,
                    "acc_ratio": ev["acc_val"] / max(none_acc, 1e-6),
                })
                print(rows[-1], flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(cfg["results_dir"], "cora_aggressive_grid.csv"), index=False)

    # Aggregate means; select: acc_ratio >= 0.95, minimize conf_tr_va (privacy proxy)
    agg = grid.groupby(["lam", "sigma"], as_index=False).mean(numeric_only=True)
    feas = agg[agg.acc_ratio >= 0.95].copy()
    if len(feas) == 0:
        feas = agg.copy()
    feas = feas.sort_values(["conf_tr_va", "acc_val"], ascending=[True, False])
    best = feas.iloc[0]
    print("SELECTED aggressive", best.to_dict(), flush=True)

    locked = {"lam": 0.5, "noise_scale": 0.35}
    aggressive = {"lam": float(best.lam), "noise_scale": float(best.sigma)}

    # --- confirm with LiRA ---
    confirm_rows = []
    for name, params in [("locked", locked), ("aggressive", aggressive)]:
        for seed in CONFIRM_SEEDS:
            data = resplit(data0.clone(), seed, **ratios)
            # none baseline LiRA once per seed shared? recompute for clarity
            from training import train_gnn
            m0 = SAGE(nf, 64, nc)
            train_gnn(m0, data, device, epochs=50)
            m0.eval()
            with torch.no_grad():
                p_none = F.softmax(m0(data.x.to(device), data.edge_index.to(device)).cpu(), 1).numpy()
            none_lira, none_tpr = _lira(
                "Cora", cfg, split_kw, {"lam": 0.0, "noise_scale": 0.0}, seed,
                p_none, data, nf, nc, device,
            ) if False else (float("nan"), float("nan"))
            # Fix: none needs shadows without sami - use zero noise none path
            shadow_p, shadow_tr, shadow_te = [], [], []
            for k in range(N_SHADOWS):
                sh_seed = seed + 999 + k * 10007
                sh, _, _ = _make_shadow_data("Cora", cfg["data_dir"], sh_seed, split_kw)
                m = SAGE(nf, 64, nc)
                train_gnn(m, sh, device, epochs=50)
                m.eval()
                with torch.no_grad():
                    sp = F.softmax(m(sh.x.to(device), sh.edge_index.to(device)).cpu(), 1).numpy()
                shadow_p.append(sp)
                shadow_tr.append(sh.train_mask.numpy())
                shadow_te.append(sh.test_mask.numpy())
            y = data.y.numpy(); tr = data.train_mask.numpy(); te = data.test_mask.numpy()
            none_lira, _, _, none_tpr = lira_gaussian_auc(
                p_none, y, tr, te, shadow_p, shadow_tr, shadow_te
            )
            none_acc = float(accuracy_score(y[te], p_none[te].argmax(1)))

            ev = _train_eval(data, nf, nc, device, params["lam"], params["noise_scale"], seed)
            sami_lira, sami_tpr = _lira(
                "Cora", cfg, split_kw, params, seed, ev["p"], data, nf, nc, device
            )
            confirm_rows.append({
                "config": name,
                "lam": params["lam"],
                "sigma": params["noise_scale"],
                "seed": seed,
                "none_acc": none_acc,
                "sami_acc": ev["acc_test"],
                "none_lira": float(none_lira),
                "sami_lira": float(sami_lira),
                "sami_tpr01": float(sami_tpr),
                "acc_ratio": ev["acc_test"] / max(none_acc, 1e-6),
            })
            print(confirm_rows[-1], flush=True)

    conf = pd.DataFrame(confirm_rows)
    conf.to_csv(os.path.join(cfg["results_dir"], "cora_aggressive_confirm.csv"), index=False)
    summary = {
        "selected": aggressive,
        "locked": locked,
        "means": conf.groupby("config")[["none_acc", "sami_acc", "none_lira", "sami_lira", "acc_ratio"]]
        .mean().round(4).to_dict(),
    }
    with open(os.path.join(cfg["results_dir"], "cora_aggressive_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
