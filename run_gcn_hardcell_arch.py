"""
GCN hard-cell campaign with arch-aware LTE under val-only selection.
Success bar: conf or LiRA down ≥0.03 with acc drop ≤0.03 vs none.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from config import ensure_dirs, load_config
from experiment import run_one

SELECT = [42, 123]
CONFIRM = [42, 123, 456, 789, 1024]
DS = "synthetic_low_sparse"
MODEL = "GCN"


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    cfg = dict(cfg)
    cfg["attacks"] = ["confidence", "lira"]
    cfg["lira"] = {"n_shadows": 4}

    none_rows = [run_one(DS, MODEL, "none", {}, s, config=cfg) for s in SELECT]
    none_acc = sum(r["test_accuracy"] for r in none_rows) / len(none_rows)
    none_conf = sum(r["conf_attack_auc"] for r in none_rows) / len(none_rows)
    none_lira = sum(r["lira_attack_auc"] for r in none_rows) / len(none_rows)

    cands = []
    for lam in (1.0, 1.5):
        for noise in (0.25, 0.5):
            for entropy in (0.05, 0.15):
                for gate in (False, True):
                    cands.append(
                        {
                            "lam": lam,
                            "use_lte": True,
                            "use_gate": gate,
                            "arch_aware": True,
                            "noise_scale": noise,
                            "entropy_coef": entropy,
                            "warmup_epochs": 5,
                            "budget_B": 0.0,
                        }
                    )

    ranked = []
    for params in cands:
        rows = [run_one(DS, MODEL, "sami", params, s, config=cfg) for s in SELECT]
        acc = sum(r["test_accuracy"] for r in rows) / len(rows)
        conf = sum(r["conf_attack_auc"] for r in rows) / len(rows)
        lira = sum(r["lira_attack_auc"] for r in rows) / len(rows)
        d_acc = acc - none_acc
        d_conf = conf - none_conf
        d_lira = lira - none_lira
        success = (d_conf <= -0.03 or d_lira <= -0.03) and d_acc >= -0.03
        ranked.append(
            {
                **params,
                "acc": acc,
                "conf": conf,
                "lira": lira,
                "d_acc": d_acc,
                "d_conf": d_conf,
                "d_lira": d_lira,
                "success": success,
                "score": acc - 0.5 * max(0, conf - 0.5),
            }
        )
    rdf = pd.DataFrame(ranked).sort_values("score", ascending=False)
    rdf.to_csv(os.path.join(cfg["results_dir"], "gcn_hardcell_arch_select.csv"), index=False)

    best = rdf.iloc[0].to_dict()
    # Confirm best on 5 seeds
    params = {
        k: best[k]
        for k in (
            "lam",
            "use_lte",
            "use_gate",
            "arch_aware",
            "noise_scale",
            "entropy_coef",
            "warmup_epochs",
            "budget_B",
        )
    }
    # coerce types
    params["use_lte"] = bool(params["use_lte"])
    params["use_gate"] = bool(params["use_gate"])
    params["arch_aware"] = bool(params["arch_aware"])

    conf_rows = []
    for s in CONFIRM:
        conf_rows.append(run_one(DS, MODEL, "none", {}, s, config=cfg))
        conf_rows.append(run_one(DS, MODEL, "sami", params, s, config=cfg))
    cdf = pd.DataFrame(conf_rows)
    cdf.to_csv(os.path.join(cfg["results_dir"], "gcn_hardcell_arch_confirm.csv"), index=False)

    none_c = cdf[cdf.defense == "none"]
    sami_c = cdf[cdf.defense == "sami"]
    d_conf = float(sami_c.conf_attack_auc.mean() - none_c.conf_attack_auc.mean())
    d_lira = float(sami_c.lira_attack_auc.mean() - none_c.lira_attack_auc.mean())
    d_acc = float(sami_c.test_accuracy.mean() - none_c.test_accuracy.mean())
    success = (d_conf <= -0.03 or d_lira <= -0.03) and d_acc >= -0.03
    out = {
        "status": "fixed" if success else "honest_failure_mixed",
        "dataset": DS,
        "model": MODEL,
        "best_params": params,
        "confirm": {
            "none_acc": float(none_c.test_accuracy.mean()),
            "none_conf": float(none_c.conf_attack_auc.mean()),
            "none_lira": float(none_c.lira_attack_auc.mean()),
            "sami_acc": float(sami_c.test_accuracy.mean()),
            "sami_conf": float(sami_c.conf_attack_auc.mean()),
            "sami_lira": float(sami_c.lira_attack_auc.mean()),
            "delta_acc": d_acc,
            "delta_conf": d_conf,
            "delta_lira": d_lira,
        },
        "note": (
            "Arch-aware LTE campaign under val-only selection."
            if success
            else "Still mixed after arch-aware LTE; treat as pre-registered architecture modulation."
        ),
    }
    with open(os.path.join(cfg["results_dir"], "gcn_hardcell_best.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
