"""
LTE quintile → membership distinguishability before/after SAMI (Cora, Actor).
Reports per-quintile conf AUROC and TPR@1%FPR on train-members stratified by LTE risk.
Also reports whether high-LTE quintiles drive the conf drop under locked SAMI.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from attacks import extract_features, tpr_at_fpr
from config import ensure_dirs, load_config
from data import load_citation, load_heterophilic, resplit
from defenses.sami import compute_lte_risk, train_gnn_sami, risk_scaled_posterior_noise
from models import SAGE
from training import train_gnn

SEEDS = [42, 123, 456]
LOCKED = {
    "lam": 0.5,
    "use_lte": True,
    "use_gate": True,
    "arch_aware": True,
    "noise_scale": 0.35,
    "budget_B": 0.0,
    "warmup_epochs": 5,
    "entropy_coef": 0.05,
}
OUT_CSV = "results/lte_quintile_attribution.csv"
OUT_JSON = "results/lte_quintile_attribution_summary.json"


def _probs_none(model, data, device):
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        return F.softmax(logits.cpu(), 1).numpy()


def _probs_sami(model, data, device, risk):
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        p = F.softmax(logits.cpu(), 1).numpy()
    # Match SAMI release: risk-scaled Laplace
    return risk_scaled_posterior_noise(
        p, risk.numpy() if hasattr(risk, "numpy") else np.asarray(risk),
        scale=float(LOCKED["noise_scale"]), seed=0
    )


def _quintile_rows(dataset, seed, defense, p, risk, y, tr, te):
    r_tr = risk[tr]
    qs = np.quantile(r_tr, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fm = extract_features(p[tr], y[tr])[:, 1]
    fn = extract_features(p[te], y[te])[:, 1]
    rows = []
    for q in range(5):
        lo, hi = qs[q], qs[q + 1]
        m = (r_tr >= lo) & (r_tr < hi) if q < 4 else (r_tr >= lo) & (r_tr <= hi)
        if int(m.sum()) < 8:
            continue
        scores = np.concatenate([fm[m], fn])
        y_mem = np.concatenate([np.ones(int(m.sum())), np.zeros(len(fn))])
        try:
            auc = float(roc_auc_score(y_mem, scores))
        except Exception:
            auc = float("nan")
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "defense": defense,
                "quintile": q + 1,
                "risk_lo": float(lo),
                "risk_hi": float(hi),
                "n_members": int(m.sum()),
                "conf_auroc": auc,
                "tpr_at_0.01": float(tpr_at_fpr(y_mem, scores, 0.01)),
                "mean_member_conf": float(fm[m].mean()),
            }
        )
    return rows


def run_dataset(name, loader, seeds=SEEDS):
    cfg = load_config()
    device = torch.device("cpu")
    rows = []
    for seed in seeds:
        data, nc, nf = loader(name, cfg["data_dir"])
        data = resplit(data, seed)
        risk = compute_lte_risk(data, arch="sage", arch_aware=True)
        y = data.y.numpy()
        tr = data.train_mask.numpy()
        te = data.test_mask.numpy()

        # none
        m0 = SAGE(nf, 64, nc)
        train_gnn(m0, data, device, epochs=50)
        p0 = _probs_none(m0, data, device)
        rows.extend(_quintile_rows(name, seed, "none", p0, risk.numpy(), y, tr, te))

        # sami (train with alignment + release)
        m1 = SAGE(nf, 64, nc)
        train_gnn_sami(
            m1,
            data,
            device,
            epochs=50,
            lam=LOCKED["lam"],
            use_lte=True,
            use_gate=True,
            arch_aware=True,
            warmup_epochs=LOCKED["warmup_epochs"],
            entropy_coef=LOCKED["entropy_coef"],
        )
        p1 = _probs_sami(m1, data, device, risk)
        rows.extend(_quintile_rows(name, seed, "sami", p1, risk.numpy(), y, tr, te))
        print(f"done {name} seed={seed}", flush=True)
    return rows


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    all_rows = []
    all_rows.extend(run_dataset("Cora", load_citation))
    all_rows.extend(run_dataset("Actor", load_heterophilic))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)

    summary = {}
    for ds in ["Cora", "Actor"]:
        sub = df[df.dataset == ds]
        piv = (
            sub.groupby(["defense", "quintile"])["conf_auroc"]
            .mean()
            .unstack("defense")
        )
        # Does Q5 drop more than Q1 under SAMI?
        if "none" in piv.columns and "sami" in piv.columns:
            delta = piv["sami"] - piv["none"]
            summary[ds] = {
                "mean_conf_auroc_by_quintile": piv.round(4).to_dict(),
                "delta_sami_minus_none": delta.round(4).to_dict(),
                "q5_vs_q1_delta": round(float(delta.get(5, np.nan) - delta.get(1, np.nan)), 4),
                "monotonic_none": bool(
                    piv["none"].is_monotonic_increasing or piv["none"].is_monotonic_decreasing
                ),
            }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
