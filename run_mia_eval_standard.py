"""
Modern MIA-eval standard (§G):
- LTE-quintile TPR@low-FPR on Cora GraphSAGE
- Label-only gap attack on headline cells
- Tuned high-utility DP-SGD baseline (kill strawman Acc≈0.19)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from attacks import confidence_attack, gap_attack, extract_features, tpr_at_fpr
from config import ensure_dirs, load_config
from data import load_citation, resplit, homophily, density
from defenses.sami import compute_lte_risk, train_gnn_sami, risk_scaled_posterior_noise
from experiment import run_one
from graph_minibatch import train_gnn_dp_fullbatch
from models import SAGE
from training import train_gnn


SEEDS = [42, 123, 456, 789, 1024]


def quintile_tpr(seed=42):
    cfg = load_config()
    data, nc, nf = load_citation("Cora", cfg["data_dir"])
    data = resplit(data, seed)
    risk = compute_lte_risk(data, arch="sage", arch_aware=True).numpy()
    model = SAGE(nf, 64, nc)
    train_gnn(model, data, torch.device("cpu"), epochs=50)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        p = F.softmax(logits, 1).numpy()
    y = data.y.numpy()
    tr = data.train_mask.numpy()
    te = data.test_mask.numpy()
    # Score: true-label confidence
    fm = extract_features(p[tr], y[tr])[:, 1]
    # Quintiles on train members by LTE risk
    r_tr = risk[tr]
    qs = np.quantile(r_tr, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    rows = []
    # Build membership scores for all train+test; stratify train members
    fn = extract_features(p[te], y[te])[:, 1]
    for q in range(5):
        lo, hi = qs[q], qs[q + 1]
        if q < 4:
            m = (r_tr >= lo) & (r_tr < hi)
        else:
            m = (r_tr >= lo) & (r_tr <= hi)
        if m.sum() < 5:
            continue
        scores = np.concatenate([fm[m], fn])
        y_mem = np.concatenate([np.ones(m.sum()), np.zeros(len(fn))])
        rows.append(
            {
                "quintile": q + 1,
                "risk_lo": float(lo),
                "risk_hi": float(hi),
                "n_members": int(m.sum()),
                "tpr_at_0.001": tpr_at_fpr(y_mem, scores, 0.001),
                "tpr_at_0.01": tpr_at_fpr(y_mem, scores, 0.01),
                "mean_member_conf": float(fm[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def gap_table(cfg):
    cells = [
        ("Cora", "GraphSAGE", "none", {}),
        ("Cora", "GraphSAGE", "sami", {"lam": 0.5, "use_lte": True, "use_gate": True, "noise_scale": 0.25}),
        ("Cora", "GraphSAGE", "gtd", {"gamma": 1.0}),
        ("Cora", "GraphSAGE", "lbp", {"scale": 0.3}),
    ]
    # Faster: confidence+gap only, no LiRA
    cfg = dict(cfg)
    cfg["attacks"] = ["confidence"]
    cfg["lira"] = {"n_shadows": 0}
    rows = []
    for ds, model, defense, params in cells:
        for seed in SEEDS[:3]:
            row = run_one(ds, model, defense, params, seed, config=cfg)
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "defense": defense,
                    "seed": seed,
                    "acc": row["test_accuracy"],
                    "conf_auc": row["conf_attack_auc"],
                    "gap_auc": row["gap_attack_auc"],
                }
            )
    return pd.DataFrame(rows)


def tuned_dp(cfg):
    """High-utility DP reference: lower noise_multiplier, more epochs."""
    data, nc, nf = load_citation("Cora", cfg["data_dir"])
    rows = []
    for seed in SEEDS[:3]:
        d = resplit(data.clone(), seed)
        model = SAGE(nf, 64, nc)
        # Tuned: noise_multiplier=0.3, epochs=80, lr=0.05 → larger ε, better Acc
        model, eps = train_gnn_dp_fullbatch(
            model,
            d,
            torch.device("cpu"),
            epochs=80,
            lr=0.05,
            max_grad_norm=1.0,
            noise_multiplier=0.3,
            delta=1e-5,
        )
        model.eval()
        with torch.no_grad():
            logits = model(d.x, d.edge_index)
            p = F.softmax(logits, 1).numpy()
            pr = logits.argmax(1).numpy()
        y = d.y.numpy()
        tr, te = d.train_mask.numpy(), d.test_mask.numpy()
        ca, _, _, _ = confidence_attack(p[tr], p[te], y[tr], y[te], random_state=seed)
        ga, _ = gap_attack(p[tr], p[te], y[tr], y[te])
        rows.append(
            {
                "dataset": "Cora",
                "model": "GraphSAGE",
                "defense": "dp_sgd_tuned",
                "seed": seed,
                "test_accuracy": float(accuracy_score(y[te], pr[te])),
                "conf_attack_auc": float(ca),
                "gap_attack_auc": float(ga),
                "dp_epsilon_approx": float(eps),
                "noise_multiplier": 0.3,
                "epochs": 80,
                "note": "High-utility DP reference (vacuous-ish ε); not strawman Acc≈0.19.",
            }
        )
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    q = quintile_tpr(42)
    q.to_csv(os.path.join(cfg["results_dir"], "lte_quintile_tpr.csv"), index=False)
    print("Quintile TPR:\n", q)

    g = gap_table(cfg)
    g.to_csv(os.path.join(cfg["results_dir"], "gap_attack_table.csv"), index=False)
    print("Gap attack:\n", g.groupby("defense")[["acc", "conf_auc", "gap_auc"]].mean())

    dp = tuned_dp(cfg)
    dp.to_csv(os.path.join(cfg["results_dir"], "dp_reference.csv"), index=False)
    # Keep legacy strawman note alongside if needed
    print("Tuned DP:\n", dp)

    summary = {
        "quintile_tpr_top_vs_bottom": {
            "top_tpr_01": float(q.iloc[-1]["tpr_at_0.01"]) if len(q) else None,
            "bottom_tpr_01": float(q.iloc[0]["tpr_at_0.01"]) if len(q) else None,
        },
        "gap_sami_vs_none": float(
            g[g.defense == "none"]["gap_auc"].mean() - g[g.defense == "sami"]["gap_auc"].mean()
        )
        if len(g)
        else None,
        "tuned_dp_mean_acc": float(dp["test_accuracy"].mean()),
        "tuned_dp_mean_eps": float(dp["dp_epsilon_approx"].mean()),
    }
    with open(os.path.join(cfg["results_dir"], "mia_eval_standard.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
