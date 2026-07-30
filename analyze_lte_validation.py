"""
LTE scientific validation + train-ratio sensitivity.
Writes:
  results/lte_correlation.csv
  results/lte_degree_strata.csv
  results/train_ratio_sensitivity.csv
  figures/fig_lte_correlation.png
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats

from config import ensure_dirs, load_config
from data import load_citation, make_synthetic, resplit, apply_split_masks
from defenses.sami import compute_lte_risk
from models import GCN, SAGE
from training import train_gnn
from attacks import extract_features


def _load(ds, seed, split_kw, data_dir):
    if ds.startswith("synthetic_"):
        parts = ds.split("_")
        return make_synthetic(homo=parts[1], dens=parts[2], seed=seed, **split_kw)
    data, nc, nf = load_citation(ds, data_dir=data_dir)
    return resplit(data, seed, **split_kw), nc, nf


def per_node_attack_score(probs, labels, train_mask, test_mask):
    """True-label confidence as a per-node membership score (threshold attack)."""
    phi = extract_features(probs, labels)
    return phi[:, 1]


def run_lte_correlation(cfg, device, out_dir, fig_dir):
    datasets = ["Cora", "Citeseer", "synthetic_low_sparse"]
    models = {"GCN": GCN, "GraphSAGE": SAGE}
    seeds = [42, 123, 456]
    split_kw = dict(
        train_ratio=float(cfg.get("split", {}).get("train_ratio", 0.4)),
        val_ratio=float(cfg.get("split", {}).get("val_ratio", 0.2)),
        test_ratio=float(cfg.get("split", {}).get("test_ratio", 0.4)),
    )
    rows = []
    strata = []
    for ds in datasets:
        for model_name, cls in models.items():
            for seed in seeds:
                data, nc, nf = _load(ds, seed, split_kw, cfg["data_dir"])
                risk = compute_lte_risk(data, uniform=False).numpy()
                model = cls(ic=nf, h=64, oc=nc).to(device)
                train_gnn(
                    model, data, device,
                    epochs=int(cfg.get("training", {}).get("epochs", 50)),
                    lr=float(cfg.get("training", {}).get("lr", 0.01)),
                    weight_decay=float(cfg.get("training", {}).get("weight_decay", 5e-4)),
                )
                model.eval()
                with torch.no_grad():
                    logits = model(data.x.to(device), data.edge_index.to(device))
                    probs = F.softmax(logits, 1).cpu().numpy()
                y = data.y.numpy()
                scores = per_node_attack_score(probs, y, data.train_mask.numpy(), data.test_mask.numpy())
                # Correlate risk with membership advantage among evaluated nodes.
                eval_mask = data.train_mask.numpy() | data.test_mask.numpy()
                mem = data.train_mask.numpy().astype(float)
                # Signed score relative to median non-member confidence.
                nm = scores[data.test_mask.numpy()]
                med = float(np.median(nm)) if len(nm) else 0.5
                adv = scores - med
                r_e = risk[eval_mask]
                adv_e = adv[eval_mask]
                mem_e = mem[eval_mask]
                # Among members, higher risk should correlate with higher confidence advantage.
                member_idx = mem_e > 0.5
                # Gap-based validation: within risk quintiles, member vs non-member φ2 gap.
                eval_idx = np.where(eval_mask)[0]
                try:
                    bins = pd.qcut(risk[eval_idx], q=min(5, len(np.unique(risk[eval_idx]))), duplicates="drop")
                except Exception:
                    bins = pd.Series(["all"] * len(eval_idx))
                gaps = []
                for b in pd.Series(bins).unique():
                    m = (pd.Series(bins).values == b)
                    nodes = eval_idx[m]
                    mem_n = nodes[mem[nodes] > 0.5]
                    nm_n = nodes[mem[nodes] < 0.5]
                    if len(mem_n) < 3 or len(nm_n) < 3:
                        continue
                    gap = float(scores[mem_n].mean() - scores[nm_n].mean())
                    gaps.append(gap)
                # Risk vs membership-informative score: AUROC of risk as a weak membership predictor among eval nodes is not the claim;
                # instead correlate risk with per-node signed residual after subtracting class-wise median.
                if member_idx.sum() > 5 and (~member_idx).sum() > 5:
                    spearman = float(stats.spearmanr(r_e[member_idx], adv_e[member_idx]).correlation)
                    pearson = float(stats.pearsonr(r_e[member_idx], adv_e[member_idx])[0])
                    mean_gap_high = float(np.mean(gaps[-1:])) if gaps else float("nan")
                    mean_gap_low = float(np.mean(gaps[:1])) if gaps else float("nan")
                else:
                    spearman = pearson = mean_gap_high = mean_gap_low = float("nan")
                rows.append({
                    "dataset": ds, "model": model_name, "seed": seed,
                    "spearman_r_vs_adv": spearman, "pearson_r_vs_adv": pearson,
                    "gap_low_risk_bin": mean_gap_low, "gap_high_risk_bin": mean_gap_high,
                    "mean_risk": float(risk.mean()),
                })
                # Degree strata
                deg = np.bincount(data.edge_index[0].numpy(), minlength=data.num_nodes)
                for node_i in np.where(eval_mask)[0]:
                    strata.append({
                        "dataset": ds, "model": model_name, "seed": seed,
                        "degree": int(deg[node_i]),
                        "risk": float(risk[node_i]),
                        "score": float(scores[node_i]),
                        "is_member": int(mem[node_i]),
                    })
    rdf = pd.DataFrame(rows)
    sdf = pd.DataFrame(strata)
    rdf.to_csv(os.path.join(out_dir, "lte_correlation.csv"), index=False)
    sdf.to_csv(os.path.join(out_dir, "lte_degree_strata.csv"), index=False)

    # Scatter: risk quintile vs member−nonmember φ2 gap on Cora GCN
    sub = sdf[(sdf.dataset == "Cora") & (sdf.model == "GCN")]
    if not sub.empty:
        sub = sub.copy()
        try:
            sub["risk_bin"] = pd.qcut(sub["risk"], q=min(5, sub["risk"].nunique()), duplicates="drop")
        except Exception:
            sub["risk_bin"] = 0
        gaps = []
        labels = []
        for i, (b, g) in enumerate(sub.groupby("risk_bin", observed=True)):
            mem = g[g.is_member == 1]["score"].mean()
            nm = g[g.is_member == 0]["score"].mean()
            gaps.append(mem - nm)
            labels.append(str(i + 1))
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(range(len(gaps)), gaps, marker="o")
        ax.set_xticks(range(len(gaps)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("LTE risk quintile")
        ax.set_ylabel("Member − non-member mean φ₂")
        ax.set_title("LTE validation: risk vs membership φ-gap (Cora GCN)")
        ax.axhline(0.0, color="gray", ls="--", lw=1)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_lte_correlation.png"), dpi=300)
        plt.close(fig)
    print(rdf.groupby(["dataset", "model"])[["spearman_r_vs_adv", "pearson_r_vs_adv"]].mean())
    return rdf, sdf


def run_train_ratio(cfg, device, out_dir):
    from experiment import run_one

    ratios = [0.2, 0.4, 0.6]
    datasets = ["Cora", "Citeseer"]
    seeds = [42, 123]
    rows = []
    for tr in ratios:
        local = dict(cfg)
        local["split"] = {"train_ratio": tr, "val_ratio": 0.2, "test_ratio": max(0.2, 1.0 - tr - 0.2)}
        local["lira"] = {"n_shadows": 2}
        local["attacks"] = ["confidence", "threshold", "lira"]
        for ds in datasets:
            for model in ["GCN", "GraphSAGE"]:
                for seed in seeds:
                    print(f"ratio={tr} {ds}/{model} seed={seed}", flush=True)
                    r = run_one(ds, model, "none", {}, seed, device=device, config=local)
                    r["train_ratio"] = tr
                    rows.append(r)
                    r2 = run_one(
                        ds, model, "sami",
                        {"lam": 0.5, "use_lte": True, "use_gate": True, "beta": 0.0,
                         "warmup_epochs": 5, "noise_scale": 0.25},
                        seed, device=device, config=local,
                    )
                    r2["train_ratio"] = tr
                    rows.append(r2)
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "train_ratio_sensitivity.csv")
    df.to_csv(path, index=False)
    print(f"Wrote {path}")
    return df


def main():
    os.environ["PRIVACYGNN_CONFIG"] = "experiment_config_confirmatory.yaml"
    cfg = load_config()
    ensure_dirs(cfg)
    device = torch.device(cfg.get("device", "cpu"))
    fig_dir = cfg["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    run_lte_correlation(cfg, device, cfg["results_dir"], fig_dir)
    run_train_ratio(cfg, device, cfg["results_dir"])


if __name__ == "__main__":
    main()
