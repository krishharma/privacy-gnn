"""
Choquette-Choo-style label-only boundary (hop-off) attack on GNN node APIs.

Black-box: adversary observes only hard argmax labels.
For each node, binary-search the L2 feature-noise scale until the predicted
label flips; score = -log(1 + sigma_flip) (smaller distance ⇒ more member-like
for overfit models). Report AUROC of scores (train members vs test non-members).

This is NOT the gap/correctness attack in attacks.gap_attack.
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
from experiment import _resplit_kwargs, _split_kwargs
from models import SAGE
from training import train_gnn

SEEDS = [42, 123, 456]
LOCKED = dict(lam=0.5, use_lte=True, use_gate=True, arch_aware=True,
              noise_scale=0.35, warmup_epochs=5, entropy_coef=0.05)
MAX_QUERIES = 16  # binary search steps per node
N_SAMPLE = 400  # subsample members + nonmembers for CPU budget


def _predict_label(model, x, edge_index, device, defense, risk=None, seed=0):
    """Hard label only (argmax). For SAMI, noise is applied to posteriors then argmax."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device), edge_index.to(device))
        if defense == "sami" and risk is not None:
            p = F.softmax(logits.cpu(), 1).numpy()
            r = risk.numpy() if hasattr(risk, "numpy") else np.asarray(risk)
            p = risk_scaled_posterior_noise(p, r, scale=LOCKED["noise_scale"], seed=seed)
            return p.argmax(1)
        return logits.argmax(1).cpu().numpy()


def hopoff_sigma(model, data, device, node_idx, defense, risk, seed, base_pred):
    """Binary search noise scale on node features until label flips."""
    x0 = data.x.clone()
    edge = data.edge_index
    rng = np.random.default_rng(seed + int(node_idx))
    # Random direction in feature space for this node
    d = rng.normal(size=x0.shape[1]).astype(np.float32)
    nrm = float(np.linalg.norm(d)) + 1e-8
    d = d / nrm

    lo, hi = 0.0, 8.0
    # Expand hi until flip or cap
    queries = 0
    flipped = False
    for _ in range(6):
        x = x0.clone()
        x[node_idx] = x0[node_idx] + hi * torch.from_numpy(d)
        pred = _predict_label(model, x, edge, device, defense, risk, seed=seed + queries)
        queries += 1
        if pred[node_idx] != base_pred:
            flipped = True
            break
        hi *= 2.0
        if hi > 64:
            break
    if not flipped:
        return hi, queries  # far from boundary / robust

    for _ in range(MAX_QUERIES - queries):
        mid = 0.5 * (lo + hi)
        x = x0.clone()
        x[node_idx] = x0[node_idx] + mid * torch.from_numpy(d)
        pred = _predict_label(model, x, edge, device, defense, risk, seed=seed + queries)
        queries += 1
        if pred[node_idx] != base_pred:
            hi = mid
        else:
            lo = mid
    return hi, queries


def run_defense(dataset, defense, seed):
    cfg = load_config()
    device = torch.device(cfg.get("device", "cpu"))
    ratios = _resplit_kwargs(_split_kwargs(cfg))
    data, nc, nf = load_citation(dataset, cfg["data_dir"])
    data = resplit(data, seed, **ratios)

    model = SAGE(nf, 64, nc)
    if defense == "none":
        train_gnn(model, data, device, epochs=50)
        risk = None
    else:
        train_gnn_sami(
            model, data, device, epochs=50,
            lam=0.5, use_lte=True, use_gate=True, arch_aware=True,
            warmup_epochs=5, entropy_coef=0.05,
        )
        risk = compute_lte_risk(data, arch="sage", arch_aware=True)

    y = data.y.numpy()
    tr = data.train_mask.numpy()
    te = data.test_mask.numpy()
    base = _predict_label(model, data.x, data.edge_index, device, defense, risk, seed)
    acc = float(accuracy_score(y[te], base[te]))

    rng = np.random.default_rng(seed)
    mem = np.where(tr)[0]
    non = np.where(te)[0]
    mem_s = rng.choice(mem, size=min(N_SAMPLE // 2, len(mem)), replace=False)
    non_s = rng.choice(non, size=min(N_SAMPLE // 2, len(non)), replace=False)

    sigmas, labels = [], []
    total_q = 0
    for v in mem_s:
        sig, q = hopoff_sigma(model, data, device, int(v), defense, risk, seed, int(base[v]))
        sigmas.append(sig)
        labels.append(1)
        total_q += q
    for v in non_s:
        sig, q = hopoff_sigma(model, data, device, int(v), defense, risk, seed, int(base[v]))
        sigmas.append(sig)
        labels.append(0)
        total_q += q

    sigmas = np.asarray(sigmas, dtype=float)
    labels = np.asarray(labels, dtype=int)
    # Closer to boundary → higher membership score
    scores = -np.log1p(sigmas)
    try:
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        auc = float("nan")
    return {
        "dataset": dataset,
        "defense": defense,
        "seed": seed,
        "acc": acc,
        "boundary_auroc": auc,
        "mean_sigma_member": float(sigmas[labels == 1].mean()),
        "mean_sigma_nonmember": float(sigmas[labels == 0].mean()),
        "n_nodes_scored": int(len(labels)),
        "mean_queries_per_node": float(total_q / max(len(labels), 1)),
    }


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    rows = []
    for defense in ["none", "sami"]:
        for seed in SEEDS:
            print(f"boundary {defense} seed={seed}", flush=True)
            rows.append(run_defense("Cora", defense, seed))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    path = os.path.join(cfg["results_dir"], "choquette_boundary_cora.csv")
    df.to_csv(path, index=False)
    summary = {
        d: {
            "acc": float(df[df.defense == d].acc.mean()),
            "boundary_auroc": float(df[df.defense == d].boundary_auroc.mean()),
            "mean_sigma_member": float(df[df.defense == d].mean_sigma_member.mean()),
            "mean_sigma_nonmember": float(df[df.defense == d].mean_sigma_nonmember.mean()),
        }
        for d in ["none", "sami"]
    }
    with open(os.path.join(cfg["results_dir"], "choquette_boundary_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
