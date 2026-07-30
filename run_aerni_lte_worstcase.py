"""
Aerni/Tramèr-style vulnerable-subset check + Spearman bootstrap CIs.

1) Top-decile LTE members vs all test non-members: conf AUROC (none vs locked SAMI).
2) Bootstrap CI on Spearman(r_v, |φ₂-gap|) under node resampling (fixed undefended model).

Not a full canary/LiRA shadow audit — population AUROC remains primary; this surfaces
whether high-LTE nodes are a worst-case pocket and whether weak ρ is stable.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from attacks import extract_features
from config import ensure_dirs, load_config
from data import load_citation, load_heterophilic, resplit
from defenses.sami import compute_lte_risk, train_gnn_sami, risk_scaled_posterior_noise
from experiment import _resplit_kwargs, _split_kwargs
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
DATASETS = [
    ("Cora", load_citation),
    ("Citeseer", load_citation),
    ("Actor", load_heterophilic),
    ("Chameleon", load_heterophilic),
]


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
    return risk_scaled_posterior_noise(
        p, risk.numpy() if hasattr(risk, "numpy") else np.asarray(risk),
        scale=float(LOCKED["noise_scale"]), seed=0,
    )


def _phi2_gap(probs, y, te):
    phi2 = probs[np.arange(len(y)), y]
    gap = np.zeros(len(y), dtype=float)
    for c in np.unique(y):
        nm = te & (y == c)
        if not nm.any():
            continue
        mu = float(phi2[nm].mean())
        gap[y == c] = phi2[y == c] - mu
    return gap, phi2


def top_decile_auroc(p, risk, y, tr, te):
    thr = np.quantile(risk[tr], 0.9)
    m = tr & (risk >= thr)
    if int(m.sum()) < 8:
        return float("nan"), int(m.sum())
    fm = extract_features(p[m], y[m])[:, 1]
    fn = extract_features(p[te], y[te])[:, 1]
    scores = np.concatenate([fm, fn])
    y_mem = np.concatenate([np.ones(len(fm)), np.zeros(len(fn))])
    try:
        return float(roc_auc_score(y_mem, scores)), int(m.sum())
    except Exception:
        return float("nan"), int(m.sum())


def spearman_bootstrap(risk, gap, mask, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.where(mask)[0]
    x = risk[idx]
    g = np.abs(gap[idx])
    rhos = []
    n = len(idx)
    for _ in range(n_boot):
        b = rng.integers(0, n, size=n)
        r, _ = spearmanr(x[b], g[b])
        if np.isfinite(r):
            rhos.append(float(r))
    rhos = np.asarray(rhos)
    point, p = spearmanr(x, g)
    return {
        "spearman": float(point),
        "pvalue": float(p),
        "ci_lo": float(np.quantile(rhos, 0.025)),
        "ci_hi": float(np.quantile(rhos, 0.975)),
        "n": int(n),
    }


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    device = torch.device(cfg.get("device", "cpu"))
    split_kw = _resplit_kwargs(_split_kwargs(cfg))
    worst_rows = []
    spear_rows = []

    for name, loader in DATASETS:
        print(f"== {name}", flush=True)
        # Spearman CI on seed 42 undefended
        data, nc, nf = loader(name, cfg["data_dir"])
        data = resplit(data, 42, **split_kw)
        m0 = SAGE(nf, 64, nc)
        train_gnn(m0, data, device, epochs=50)
        p0 = _probs_none(m0, data, device)
        risk = compute_lte_risk(data, arch="sage", arch_aware=True).cpu().numpy()
        y = data.y.cpu().numpy()
        tr = data.train_mask.cpu().numpy()
        te = data.test_mask.cpu().numpy()
        gap, _ = _phi2_gap(p0, y, te)
        sp = spearman_bootstrap(risk, gap, tr | te)
        sp["dataset"] = name
        spear_rows.append(sp)
        print("  spearman", sp, flush=True)

        for seed in SEEDS:
            data, nc, nf = loader(name, cfg["data_dir"])
            data = resplit(data, seed, **split_kw)
            risk = compute_lte_risk(data, arch="sage", arch_aware=True)
            y = data.y.numpy()
            tr = data.train_mask.numpy()
            te = data.test_mask.numpy()
            rnp = risk.numpy()

            m_none = SAGE(nf, 64, nc)
            train_gnn(m_none, data, device, epochs=50)
            p_none = _probs_none(m_none, data, device)
            auc0, n0 = top_decile_auroc(p_none, rnp, y, tr, te)

            m_sami = SAGE(nf, 64, nc)
            train_gnn_sami(
                m_sami, data, device, epochs=50,
                lam=LOCKED["lam"], use_lte=True, use_gate=True,
                arch_aware=True, warmup_epochs=5, entropy_coef=0.05,
            )
            p_sami = _probs_sami(m_sami, data, device, risk)
            auc1, n1 = top_decile_auroc(p_sami, rnp, y, tr, te)

            worst_rows.append({
                "dataset": name, "seed": seed,
                "none_topdecile_conf_auroc": auc0,
                "sami_topdecile_conf_auroc": auc1,
                "delta": auc1 - auc0 if np.isfinite(auc0) and np.isfinite(auc1) else float("nan"),
                "n_topdecile_members": n0,
            })
            print(f"  seed {seed}: none={auc0:.3f} sami={auc1:.3f} Δ={auc1-auc0:+.3f}", flush=True)

    worst = pd.DataFrame(worst_rows)
    spear = pd.DataFrame(spear_rows)
    out_dir = cfg["results_dir"]
    worst.to_csv(os.path.join(out_dir, "aerni_lte_topdecile.csv"), index=False)
    spear.to_csv(os.path.join(out_dir, "lte_spearman_bootstrap_ci.csv"), index=False)

    summary = {
        "topdecile_means": {
            ds: {
                k: float(v)
                for k, v in worst[worst.dataset == ds][
                    ["none_topdecile_conf_auroc", "sami_topdecile_conf_auroc", "delta"]
                ].mean().items()
            }
            for ds in worst.dataset.unique()
        },
        "spearman_ci": spear.set_index("dataset").to_dict(orient="index"),
        "note": (
            "Vulnerable-subset = top-decile LTE train members vs all test non-members; "
            "conf AUROC (not full shadow LiRA). Spearman CI = node bootstrap on one "
            "undefended GraphSAGE (seed 42)."
        ),
    }
    with open(os.path.join(out_dir, "aerni_lte_worstcase_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
