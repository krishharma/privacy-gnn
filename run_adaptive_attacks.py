"""
Adaptive attacker stress tests:
  1) Multi-query Laplace averaging (K in {1,5,20}) against SAMI release.
  2) MLP-φ attacker (already wired into run_one).
  3) Defense-aware shadows are the default in experiment.run_one.

Writes results/adaptive_multiquery.csv and figures/fig_multiquery.png.
"""
from __future__ import annotations

import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from config import ensure_dirs, load_config
from experiment import run_one

CELLS = [
    ("Cora", "GraphSAGE"),
    ("Citeseer", "GraphSAGE"),
    ("synthetic_low_sparse", "GCN"),
]
SEEDS = [42, 123, 456]
K_VALUES = [1, 5, 20]
SAMI = {"lam": 0.5, "use_lte": True, "use_gate": True, "beta": 0.0, "warmup_epochs": 5, "noise_scale": 0.35}


def main():
    os.environ["PRIVACYGNN_CONFIG"] = "experiment_config_confirmatory.yaml"
    cfg = load_config()
    ensure_dirs(cfg)
    cfg["lira"] = {"n_shadows": 4}
    cfg["attacks"] = ["confidence", "threshold", "shadow", "lira", "mlp_phi"]
    cfg["run_mlp_phi_attack"] = True
    device = torch.device(cfg.get("device", "cpu"))

    rows = []
    for ds, model in CELLS:
        for seed in SEEDS:
            print(f"BASE none {ds}/{model} seed={seed}", flush=True)
            r0 = run_one(ds, model, "none", {}, seed, device=device, config=cfg)
            r0["attack_setting"] = "none_k1"
            rows.append(r0)
            for k in K_VALUES:
                local = dict(cfg)
                local["multi_query_k"] = k
                print(f"SAMI multi-query K={k} {ds}/{model} seed={seed}", flush=True)
                r = run_one(ds, model, "sami", SAMI, seed, device=device, config=local)
                r["attack_setting"] = f"sami_k{k}"
                rows.append(r)

    df = pd.DataFrame(rows)
    out = os.path.join(cfg["results_dir"], "adaptive_multiquery.csv")
    df.to_csv(out, index=False)
    print(df.groupby(["dataset", "model", "attack_setting"])[
        ["test_accuracy", "conf_attack_auc", "lira_attack_auc", "mlp_phi_attack_auc"]
    ].mean())

    # Figure: conf AUROC vs K for Cora GraphSAGE
    sub = df[(df.dataset == "Cora") & (df.model == "GraphSAGE") & (df.defense == "sami")]
    if not sub.empty:
        g = sub.groupby("multi_query_k")[["conf_attack_auc", "lira_attack_auc"]].mean()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(g.index, g["conf_attack_auc"], marker="o", label="conf AUROC")
        ax.plot(g.index, g["lira_attack_auc"], marker="s", label="LiRA AUROC")
        none_ca = df[(df.dataset == "Cora") & (df.model == "GraphSAGE") & (df.defense == "none")][
            "conf_attack_auc"
        ].mean()
        ax.axhline(none_ca, color="gray", linestyle="--", label="none conf")
        ax.set_xlabel("Multi-query averaging K")
        ax.set_ylabel("Attack AUROC")
        ax.set_title("Adaptive multi-query vs SAMI (Cora GraphSAGE)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cfg["figures_dir"], "fig_multiquery.png"), dpi=300)
        plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
