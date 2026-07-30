"""
Canary / vulnerable-subset LiRA (Aerni-style, full shadow LiRA).

Reports for Cora GraphSAGE (none vs locked SAMI):
  1) Population LiRA (train vs test)
  2) Top-decile LTE members vs all test non-members (worst-case pocket)
  3) Planted canaries: K nodes with unique feature spikes, always in train,
     scored vs random test non-members

Uses n_shadows=4, 3 seeds by default (CPU-feasible).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

from config import ensure_dirs, load_config
from data import load_citation, resplit
from defenses.sami import compute_lte_risk, train_gnn_sami, risk_scaled_posterior_noise
from experiment import _resplit_kwargs, _split_kwargs, _make_shadow_data
from lira_attack import lira_gaussian_auc, lira_gaussian_scores, lira_auc_on_subset
from models import SAGE
from training import train_gnn

SEEDS = [42, 123, 456]
N_SHADOWS = 4
N_CANARIES = 64
LOCKED = {
    "lam": 0.5,
    "use_lte": True,
    "use_gate": True,
    "arch_aware": True,
    "noise_scale": 0.35,
    "warmup_epochs": 5,
    "entropy_coef": 0.05,
}


def _predict(model, data, device, defense, risk=None, seed=0):
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        p = F.softmax(logits.cpu(), 1).numpy()
    if defense == "sami" and risk is not None:
        r = risk.numpy() if hasattr(risk, "numpy") else np.asarray(risk)
        p = risk_scaled_posterior_noise(p, r, scale=LOCKED["noise_scale"], seed=seed)
    return p


def _train_target(defense, data, nf, nc, device):
    model = SAGE(nf, 64, nc)
    if defense == "none":
        train_gnn(model, data, device, epochs=50)
        risk = compute_lte_risk(data, arch="sage", arch_aware=True)
        return model, risk
    train_gnn_sami(
        model, data, device, epochs=50,
        lam=LOCKED["lam"], use_lte=True, use_gate=True,
        arch_aware=True, warmup_epochs=5, entropy_coef=0.05,
    )
    risk = compute_lte_risk(data, arch="sage", arch_aware=True)
    return model, risk


def _plant_canaries(data, seed, k=N_CANARIES):
    """Force K train nodes; spike a reserved feature channel so they are unique."""
    d = data.clone()
    rng = np.random.default_rng(seed)
    n, f = d.x.shape
    # Prefer existing train nodes; fall back to any
    tr = d.train_mask.numpy()
    pool = np.where(tr)[0]
    if len(pool) < k:
        pool = np.arange(n)
    canaries = rng.choice(pool, size=min(k, len(pool)), replace=False)
    # Spike feature 0 with unique large values
    x = d.x.clone()
    for i, v in enumerate(canaries):
        x[v, 0] = 50.0 + float(i)
    d.x = x
    # Ensure canaries are members
    tm = d.train_mask.clone()
    vm = d.val_mask.clone()
    sm = d.test_mask.clone()
    tm[:] = False
    # Rebuild: canaries + rest of original train (minus canaries already)
    tm[canaries] = True
    # keep other train nodes that aren't canaries
    other = np.where(tr)[0]
    other = other[~np.isin(other, canaries)]
    tm[other] = True
    d.train_mask = tm
    d.val_mask = vm
    d.test_mask = sm
    return d, canaries


def _shadow_bundle(dataset, data_dir, split_kw, defense, nf, nc, device, seed, plant=None):
    shadow_p, shadow_tr, shadow_te = [], [], []
    for k in range(N_SHADOWS):
        sh_seed = seed + 999 + k * 10007
        sh = _make_shadow_data(dataset, data_dir, sh_seed, {**split_kw, "protocol": "random_ratio"})
        # _make_shadow_data expects full split_kw with protocol - check
        if plant is not None:
            # Re-plant same canary indices if still valid
            sh, _ = _plant_canaries(sh, seed, k=N_CANARIES)
        m, risk = _train_target(defense, sh, nf, nc, device)
        shadow_p.append(_predict(m, sh, device, defense, risk, seed=sh_seed))
        shadow_tr.append(sh.train_mask.numpy())
        shadow_te.append(sh.test_mask.numpy())
    return shadow_p, shadow_tr, shadow_te


def run_cell(dataset, defense, seed, plant_canaries=False):
    cfg = load_config()
    device = torch.device(cfg.get("device", "cpu"))
    split_kw = _split_kwargs(cfg)
    ratios = _resplit_kwargs(split_kw)
    data, nc, nf = load_citation(dataset, cfg["data_dir"])
    data = resplit(data, seed, **ratios)
    canaries = None
    if plant_canaries:
        data, canaries = _plant_canaries(data, seed)

    model, risk = _train_target(defense, data, nf, nc, device)
    p = _predict(model, data, device, defense, risk, seed=seed)
    y = data.y.numpy()
    tr = data.train_mask.numpy()
    te = data.test_mask.numpy()
    acc = float(accuracy_score(y[te], p[te].argmax(1)))

    shadow_p, shadow_tr, shadow_te = [], [], []
    for k in range(N_SHADOWS):
        sh_seed = seed + 999 + k * 10007
        sh, _, _ = _make_shadow_data(dataset, cfg["data_dir"], sh_seed, split_kw)
        if plant_canaries:
            sh, _ = _plant_canaries(sh, seed)
        m_s, r_s = _train_target(defense, sh, nf, nc, device)
        shadow_p.append(_predict(m_s, sh, device, defense, r_s, seed=sh_seed))
        shadow_tr.append(sh.train_mask.numpy())
        shadow_te.append(sh.test_mask.numpy())

    pop_auc, _, _, pop_tpr = lira_gaussian_auc(
        p, y, tr, te, shadow_p, shadow_tr, shadow_te
    )
    scores, y_mem, _ = lira_gaussian_scores(
        p, y, tr, te, shadow_p, shadow_tr, shadow_te
    )

    rnp = risk.numpy() if hasattr(risk, "numpy") else np.asarray(risk)
    thr = np.quantile(rnp[tr], 0.9)
    top = np.where(tr & (rnp >= thr))[0]
    non = np.where(te)[0]
    top_auc, n_top, n_non = lira_auc_on_subset(scores, y_mem, top, non)

    out = {
        "dataset": dataset,
        "defense": defense,
        "seed": seed,
        "plant_canaries": plant_canaries,
        "acc": acc,
        "pop_lira": float(pop_auc),
        "pop_tpr01": float(pop_tpr),
        "topdecile_lira": float(top_auc) if top_auc == top_auc else float("nan"),
        "n_topdecile": int(n_top),
        "n_shadows": N_SHADOWS,
    }
    if canaries is not None:
        can_m = np.asarray(canaries)[tr[np.asarray(canaries)]]
        can_auc, n_c, _ = lira_auc_on_subset(scores, y_mem, can_m, non)
        out["canary_lira"] = float(can_auc) if can_auc == can_auc else float("nan")
        out["n_canaries"] = int(n_c)
    return out


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    rows = []
    # Main: population + top-decile LiRA
    for defense in ["none", "sami"]:
        for seed in SEEDS:
            print(f"canary-LiRA {defense} seed={seed}", flush=True)
            rows.append(run_cell("Cora", defense, seed, plant_canaries=False))
            print(rows[-1], flush=True)
    # Planted canaries (separate runs)
    for defense in ["none", "sami"]:
        for seed in SEEDS:
            print(f"planted-canary {defense} seed={seed}", flush=True)
            rows.append(run_cell("Cora", defense, seed, plant_canaries=True))
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(cfg["results_dir"], "canary_lira_cora.csv")
    df.to_csv(out, index=False)

    summary = {}
    for plant in [False, True]:
        sub = df[df.plant_canaries == plant]
        key = "planted" if plant else "natural_topdecile"
        summary[key] = {}
        for d in ["none", "sami"]:
            s = sub[sub.defense == d]
            summary[key][d] = {
                "acc": float(s.acc.mean()),
                "pop_lira": float(s.pop_lira.mean()),
                "topdecile_lira": float(s.topdecile_lira.mean()) if "topdecile_lira" in s else None,
                "canary_lira": float(s.canary_lira.mean()) if plant and "canary_lira" in s.columns else None,
            }
    with open(os.path.join(cfg["results_dir"], "canary_lira_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
