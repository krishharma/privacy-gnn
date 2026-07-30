"""
Pareto sweep (λ / noise), ROC overlay, timing table, DP/MaskArmor reference points.
Writes:
  results/pareto_sweep.csv
  results/timing_overhead.csv
  results/baselines_extra.csv
  figures/fig_pareto_frontier.png
  figures/fig_roc_cora_sage.png
"""
from __future__ import annotations

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import ensure_dirs, load_config
from experiment import run_one, _train_and_predict_gnn, _load_target_data, _split_kwargs
from attacks import roc_curve_points
from data import load_citation, resplit
from models import GCN, SAGE, GatedGCN, GatedSAGE
from defenses.sami import train_gnn_sami, compute_lte_risk, risk_scaled_posterior_noise
from training import train_gnn

SEEDS = [42, 123, 456]
CORA_SAGE = ("Cora", "GraphSAGE")


def pareto_sweep(cfg, device):
    rows = []
    # SAMI λ × noise on Cora GraphSAGE + hard cell
    for ds, model in [("Cora", "GraphSAGE"), ("synthetic_low_sparse", "GCN")]:
        for seed in SEEDS:
            rows.append(run_one(ds, model, "none", {}, seed, device=device, config=cfg))
            for lam in [0.1, 0.25, 0.5, 1.0]:
                for noise in [0.0, 0.1, 0.25, 0.4]:
                    dp = {
                        "lam": lam, "use_lte": True, "use_gate": True, "beta": 0.0,
                        "warmup_epochs": 5, "noise_scale": noise,
                    }
                    print(f"PARETO {ds}/{model} lam={lam} noise={noise} seed={seed}", flush=True)
                    r = run_one(ds, model, "sami", dp, seed, device=device, config=cfg)
                    r["pareto_lam"] = lam
                    r["pareto_noise"] = noise
                    rows.append(r)
            for scale in [0.1, 0.3, 0.5]:
                r = run_one(ds, model, "lbp", {"scale": scale}, seed, device=device, config=cfg)
                r["pareto_lbp_scale"] = scale
                rows.append(r)
            for gamma in [0.5, 1.0, 2.0]:
                r = run_one(
                    ds, model, "gtd",
                    {"gamma": gamma, "stage1_frac": 0.5, "pseudo_conf": 0.8},
                    seed, device=device, config=cfg,
                )
                r["pareto_gtd_gamma"] = gamma
                rows.append(r)
    return pd.DataFrame(rows)


def roc_overlay(cfg, device, fig_dir):
    """Overlay ROC for Cora GraphSAGE: none / sami / lbp / label_smoothing."""
    ds, model = CORA_SAGE
    seed = 42
    defenses = [
        ("none", {}),
        ("sami", {"lam": 0.5, "use_lte": True, "use_gate": True, "beta": 0.0, "warmup_epochs": 5, "noise_scale": 0.25}),
        ("lbp", {"scale": 0.3}),
        ("label_smoothing", {"alpha": 0.1}),
    ]
    # Re-run with storing probs via a lightweight local train
    from experiment import run_one as _  # noqa — use dedicated path below
    split_kw = _split_kwargs(cfg)
    data, nc, nf = _load_target_data(ds, cfg["data_dir"], seed, False, split_kw)
    ep = int(cfg.get("training", {}).get("epochs", 50))
    lr = float(cfg.get("training", {}).get("lr", 0.01))
    wd = float(cfg.get("training", {}).get("weight_decay", 5e-4))
    fig, ax = plt.subplots(figsize=(5, 4))
    for dn, dp in defenses:
        tk = {"dropedge_rate": 0.0, "label_smoothing": 0.0, "early_stop_patience": None, "edge_sparsify_rate": 0.0}
        if dn == "label_smoothing":
            tk["label_smoothing"] = dp.get("alpha", 0.1)
        cmk = None
        p, pr, _, _, _, _ = _train_and_predict_gnn(
            model, dn, dp, data, nf, nc, device, ep, lr, wd, tk, cmk,
            False, 1024, [15, 10], cfg, release_seed=seed, multi_query_k=1,
        )
        trm = data.train_mask.numpy()
        tem = data.test_mask.numpy()
        y = data.y.numpy()
        fpr, tpr, auc = roc_curve_points(p[trm], p[tem], y[trm], y[tem])
        ax.plot(fpr, tpr, label=f"{dn} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC overlay: Cora GraphSAGE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig_roc_cora_sage.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def timing_overhead(cfg, device, out_dir):
    """Wall-clock / peak RSS proxy for LTE+HCAG+release vs vanilla on PubMed (largest citation)."""
    import tracemalloc

    rows = []
    ds = "PubMed"
    model = "GraphSAGE"
    seed = 42
    for dn, dp in [
        ("none", {}),
        ("sami", {"lam": 0.5, "use_lte": True, "use_gate": True, "beta": 0.0, "warmup_epochs": 5, "noise_scale": 0.25}),
        ("sami_no_gate", {"lam": 0.5, "use_lte": True, "use_gate": False, "beta": 0.0, "warmup_epochs": 5, "noise_scale": 0.25}),
    ]:
        tracemalloc.start()
        t0 = time.time()
        local = dict(cfg)
        local["lira"] = {"n_shadows": 0}
        local["attacks"] = ["confidence"]
        r = run_one(ds, model, dn, dp, seed, device=device, config=local)
        wall = time.time() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({
            "dataset": ds, "model": model, "defense": dn,
            "wall_seconds": wall,
            "train_seconds": r.get("train_seconds"),
            "peak_python_mb": peak / (1024 * 1024),
            "test_accuracy": r.get("test_accuracy"),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "timing_overhead.csv")
    df.to_csv(path, index=False)
    print(df)
    return df


def baselines_extra(cfg, device, out_dir):
    """MaskArmor + DP-SGD reference on Cora GraphSAGE."""
    rows = []
    ds, model = "Cora", "GraphSAGE"
    for seed in SEEDS:
        rows.append(run_one(ds, model, "none", {}, seed, device=device, config=cfg))
        rows.append(run_one(ds, model, "maskarmor", {"top_k": 1}, seed, device=device, config=cfg))
        rows.append(run_one(ds, model, "confidence_masking", {"top_k": 2}, seed, device=device, config=cfg))
        # DP reference (may be slow / approximate)
        try:
            local = dict(cfg)
            local["dp_sgd"] = {
                "epochs": 30, "lr": 0.05, "batch_size": 512,
                "max_grad_norm": 1.0, "noise_multiplier": 1.0, "delta": 1e-5,
            }
            print(f"DP-SGD seed={seed}", flush=True)
            rows.append(run_one(ds, model, "dp_sgd", {}, seed, device=device, config=local))
        except Exception as e:
            print(f"DP-SGD skipped: {e}")
    # Actor heterophilic stretch
    for seed in SEEDS:
        try:
            print(f"Actor GraphSAGE seed={seed}", flush=True)
            rows.append(run_one("Actor", "GraphSAGE", "none", {}, seed, device=device, config=cfg))
            rows.append(run_one(
                "Actor", "GraphSAGE", "sami",
                {"lam": 0.5, "use_lte": True, "use_gate": True, "beta": 0.0, "warmup_epochs": 5, "noise_scale": 0.25},
                seed, device=device, config=cfg,
            ))
        except Exception as e:
            print(f"Actor skipped: {e}")
            with open(os.path.join(out_dir, "scaling_limitations.json"), "w") as f:
                json.dump({
                    "ogbn_arxiv": "Not bundled; MINIBATCH_DATASETS empty in ogb_loader.py. "
                                  "Documented as Volume scaling limitation for BigData audience.",
                    "actor_error": str(e),
                }, indent=2)
            break
    else:
        with open(os.path.join(out_dir, "scaling_limitations.json"), "w") as f:
            json.dump({
                "ogbn_arxiv": "Not enabled in this checkout (ogb_loader stub). "
                              "Largest measured graph: PubMed (+ Actor if downloaded). "
                              "See timing_overhead.csv for wall-clock/memory.",
            }, indent=2)
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "baselines_extra.csv")
    df.to_csv(path, index=False)
    return df


def plot_pareto(df, fig_dir):
    sub = df[(df.dataset == "Cora") & (df.model == "GraphSAGE")]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for defense, marker in [("none", "x"), ("sami", "o"), ("lbp", "s"), ("gtd", "^")]:
        g = sub[sub.defense == defense]
        if g.empty:
            continue
        ax.scatter(g["conf_attack_auc"], g["test_accuracy"], label=defense, marker=marker, alpha=0.75)
    ax.set_xlabel("Confidence-attack AUROC (lower better)")
    ax.set_ylabel("Test accuracy (higher better)")
    ax.set_title("Privacy–utility Pareto (Cora GraphSAGE)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_pareto_frontier.png"), dpi=300)
    plt.close(fig)


def main():
    os.environ["PRIVACYGNN_CONFIG"] = "experiment_config_confirmatory.yaml"
    cfg = load_config()
    ensure_dirs(cfg)
    cfg["lira"] = {"n_shadows": 2}
    cfg["attacks"] = ["confidence", "threshold", "lira"]
    cfg["run_mlp_phi_attack"] = True
    device = torch.device(cfg.get("device", "cpu"))
    fig_dir = cfg["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    print("=== ROC overlay ===", flush=True)
    roc_overlay(cfg, device, fig_dir)

    print("=== Timing ===", flush=True)
    timing_overhead(cfg, device, cfg["results_dir"])

    print("=== Baselines + Actor ===", flush=True)
    baselines_extra(cfg, device, cfg["results_dir"])

    print("=== Pareto sweep ===", flush=True)
    pdf = pareto_sweep(cfg, device)
    pout = os.path.join(cfg["results_dir"], "pareto_sweep.csv")
    pdf.to_csv(pout, index=False)
    plot_pareto(pdf, fig_dir)
    print(f"Wrote {pout}")


if __name__ == "__main__":
    main()
