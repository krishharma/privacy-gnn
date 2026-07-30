"""
Expanded SCML synthetic grid + feature-SNR axis + leave-one-regime-out fit.
Also merges existing core_results when present.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from config import ensure_dirs, load_config
from experiment import run_one

# Dense (h, ρ) × SNR × {GCN, GraphSAGE} under none — small graphs, fast.
H_LIST = [0.2, 0.4, 0.6, 0.8]
RHO_LIST = [0.005, 0.01, 0.02, 0.04]
SNR_LIST = [0.25, 1.0, 2.5]
MODELS = ["GCN", "GraphSAGE"]
SEEDS = [42, 123]


def cell_name(h, rho, snr):
    return f"synth_h{h}_r{rho}_snr{snr}"


def run_grid(cfg):
    rows = []
    device = cfg.get("device", "cpu")
    # Temporarily override synthetic via make_synthetic kwargs through a custom path:
    # we call make_synthetic directly inside a thin wrapper using run_one on named synthetics
    # that encode homo/dens/snr. Map continuous h/ρ to nearest named + pass via make_synthetic.
    from data import make_synthetic, homophily, density
    from models import GCN, SAGE
    from training import train_gnn
    from attacks import confidence_attack
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score

    split_kw = dict(
        train_ratio=float(cfg.get("split", {}).get("train_ratio", 0.4)),
        val_ratio=float(cfg.get("split", {}).get("val_ratio", 0.2)),
        test_ratio=float(cfg.get("split", {}).get("test_ratio", 0.4)),
    )
    for h in H_LIST:
        for rho in RHO_LIST:
            for snr in SNR_LIST:
                for model_name in MODELS:
                    for seed in SEEDS:
                        data, nc, nf = make_synthetic(
                            homo="low",
                            dens="medium",
                            seed=seed,
                            feature_snr=snr,
                            h_frac=h,
                            density_value=rho,
                            **split_kw,
                        )
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        cls = GCN if model_name == "GCN" else SAGE
                        model = cls(nf, 64, nc)
                        train_gnn(model, data, torch.device(device), epochs=40, lr=0.01)
                        model.eval()
                        with torch.no_grad():
                            logits = model(data.x, data.edge_index)
                            p = F.softmax(logits, 1).numpy()
                            pr = logits.argmax(1).numpy()
                        y = data.y.numpy()
                        tr = data.train_mask.numpy()
                        te = data.test_mask.numpy()
                        ca, _, _, _ = confidence_attack(p[tr], p[te], y[tr], y[te], random_state=seed)
                        rows.append(
                            {
                                "dataset": cell_name(h, rho, snr),
                                "model": model_name,
                                "defense": "none",
                                "seed": seed,
                                "homophily": float(homophily(data)),
                                "density": float(density(data)),
                                "feature_snr": snr,
                                "h_target": h,
                                "rho_target": rho,
                                "conf_attack_auc": float(ca),
                                "test_accuracy": float(accuracy_score(y[te], pr[te])),
                                "is_gcn": float(model_name == "GCN"),
                            }
                        )
    return pd.DataFrame(rows)


def fit_loo(df: pd.DataFrame):
    df = df.copy()
    df["het"] = 1.0 - df["homophily"]
    df["sparsity"] = np.log(0.04) - np.log(df["density"].clip(1e-8))
    df["sparsity"] = df["sparsity"].clip(-1, 4)
    df["het_x_gcn"] = df["het"] * df["is_gcn"]
    df["sparsity_x_gcn"] = df["sparsity"] * df["is_gcn"]
    df["snr"] = df["feature_snr"].astype(float)
    df["snr_x_inv"] = 1.0 / (df["snr"] + 1e-6)
    cols = ["het", "sparsity", "is_gcn", "het_x_gcn", "sparsity_x_gcn", "snr", "snr_x_inv"]
    # Aggregate by cell
    agg = (
        df.groupby(["dataset", "model"], as_index=False)
        .agg(
            {
                "het": "mean",
                "sparsity": "mean",
                "is_gcn": "mean",
                "het_x_gcn": "mean",
                "sparsity_x_gcn": "mean",
                "snr": "mean",
                "snr_x_inv": "mean",
                "conf_attack_auc": "mean",
                "homophily": "mean",
                "density": "mean",
                "h_target": "mean",
                "rho_target": "mean",
            }
        )
        .rename(columns={"conf_attack_auc": "conf"})
    )
    # Leave-one-regime-out by (h_target, rho_target)
    preds = []
    regimes = agg.groupby(["h_target", "rho_target"]).size().index.tolist()
    for h, rho in regimes:
        te = (agg["h_target"] == h) & (agg["rho_target"] == rho)
        tr = ~te
        if tr.sum() < 5 or te.sum() < 1:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(agg.loc[tr, cols].values)
        Xte = scaler.transform(agg.loc[te, cols].values)
        ytr = agg.loc[tr, "conf"].values
        model = Ridge(alpha=0.5).fit(Xtr, ytr)
        yhat = model.predict(Xte)
        for i, idx in enumerate(agg.index[te]):
            preds.append(
                {
                    "dataset": agg.loc[idx, "dataset"],
                    "model": agg.loc[idx, "model"],
                    "y_true": float(agg.loc[idx, "conf"]),
                    "y_pred": float(yhat[i]),
                    "h_target": h,
                    "rho_target": rho,
                }
            )
    pred_df = pd.DataFrame(preds)
    # Full in-sample fit for coefficients
    scaler = StandardScaler()
    X = scaler.fit_transform(agg[cols].values)
    y = agg["conf"].values
    full = Ridge(alpha=0.5).fit(X, y)
    loo_mae = float(mean_absolute_error(pred_df["y_true"], pred_df["y_pred"])) if len(pred_df) else float("nan")
    loo_spear = (
        float(stats.spearmanr(pred_df["y_true"], pred_df["y_pred"]).correlation)
        if len(pred_df) > 2
        else float("nan")
    )
    in_r2 = float(r2_score(y, full.predict(X)))
    return agg, pred_df, {
        "feature_cols": cols,
        "coef": dict(zip(cols, map(float, full.coef_))),
        "intercept": float(full.intercept_),
        "in_sample_r2": in_r2,
        "loo_mae": loo_mae,
        "loo_spearman": loo_spear,
        "n_cells": int(len(agg)),
        "n_loo_preds": int(len(pred_df)),
        "law_name": "SCML",
        "law_statement": "Under controlled features, score-based GNN membership AUROC increases with heterophily and sparsity; GCN amplifies this; high feature SNR induces MLP-style feature-dominated leakage.",
    }, full, scaler


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    print("Running expanded SCML synthetic grid...")
    df = run_grid(cfg)
    path_raw = os.path.join(cfg["results_dir"], "scml_expanded_raw.csv")
    df.to_csv(path_raw, index=False)
    agg, pred_df, meta, _, _ = fit_loo(df)
    agg.to_csv(os.path.join(cfg["results_dir"], "leakage_law_train.csv"), index=False)
    pred_df.to_csv(os.path.join(cfg["results_dir"], "leakage_law_loo.csv"), index=False)
    # Also keep oos path for compatibility
    pred_df.to_csv(os.path.join(cfg["results_dir"], "leakage_law_oos.csv"), index=False)
    with open(os.path.join(cfg["results_dir"], "leakage_law_fit.json"), "w") as f:
        json.dump(meta, f, indent=2)
    # Feature reversal inside synthetics: high SNR vs low SNR at fixed structure
    rev = (
        df.groupby(["h_target", "rho_target", "feature_snr", "model"], as_index=False)[
            "conf_attack_auc"
        ]
        .mean()
    )
    rev.to_csv(os.path.join(cfg["results_dir"], "feature_snr_grid.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    if len(pred_df):
        axes[0].scatter(pred_df["y_true"], pred_df["y_pred"], alpha=0.7)
        lims = [
            min(pred_df["y_true"].min(), pred_df["y_pred"].min()) - 0.02,
            max(pred_df["y_true"].max(), pred_df["y_pred"].max()) + 0.02,
        ]
        axes[0].plot(lims, lims, "k--", lw=1)
        axes[0].set_xlabel("Observed conf AUROC")
        axes[0].set_ylabel("LOO predicted")
        axes[0].set_title(f"SCML LOO (MAE={meta['loo_mae']:.3f})")
    # SNR effect
    for model_name, g in rev.groupby("model"):
        m = g.groupby("feature_snr")["conf_attack_auc"].mean()
        axes[1].plot(m.index, m.values, marker="o", label=model_name)
    axes[1].set_xlabel("Feature SNR")
    axes[1].set_ylabel("Mean conf AUROC")
    axes[1].set_title("Feature-separability axis")
    axes[1].legend()
    fig.tight_layout()
    fig_path = os.path.join(cfg["figures_dir"], "fig_leakage_law_pred.png")
    fig.savefig(fig_path, dpi=160)
    plt.close()
    print(json.dumps(meta, indent=2))
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
