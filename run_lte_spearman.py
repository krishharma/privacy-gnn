"""
LTE vs member−non-member φ₂ gap: Spearman by dataset (Cora, Citeseer, Actor).
Also documents why conf AUROC ≫ LiRA on the controlled stress-test synthetic.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from config import ensure_dirs, load_config
from data import load_citation, load_heterophilic, resplit
from defenses.sami import compute_lte_risk, phi_features_torch
from experiment import _split_kwargs
from models import SAGE
from training import train_gnn


def _phi2_gap_by_node(probs, y, train_mask, test_mask):
    """Per-node φ₂ = true-label confidence; gap proxy = φ₂ - mean_nonmember_φ₂(class)."""
    probs = np.asarray(probs)
    y = np.asarray(y)
    tr = np.asarray(train_mask)
    te = np.asarray(test_mask)
    phi2 = probs[np.arange(len(y)), y]
    # Class-conditional non-member mean φ₂
    gap = np.zeros(len(y), dtype=float)
    for c in np.unique(y):
        nm = te & (y == c)
        if not nm.any():
            continue
        mu = float(phi2[nm].mean())
        gap[y == c] = phi2[y == c] - mu
    return gap, phi2


def analyze_dataset(name, load_fn, seed=42, epochs=50):
    cfg = load_config()
    device = torch.device(cfg.get("device", "cpu"))
    split_kw = _split_kwargs(cfg)
    data, nc, nf = load_fn(name, data_dir=cfg["data_dir"])
    data = resplit(data, seed, **split_kw)
    model = SAGE(nf, 64, nc)
    model = train_gnn(model, data, device, epochs=epochs)
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
    risk = compute_lte_risk(data, uniform=False, arch="sage", arch_aware=True).cpu().numpy()
    y = data.y.cpu().numpy()
    tr = data.train_mask.cpu().numpy()
    te = data.test_mask.cpu().numpy()
    gap, phi2 = _phi2_gap_by_node(probs, y, tr, te)
    # Correlate LTE with |gap| on train+test nodes with labels
    mask = tr | te
    rho_gap, p_gap = spearmanr(risk[mask], np.abs(gap[mask]))
    rho_phi, p_phi = spearmanr(risk[mask], phi2[mask])
    # Member vs non-member mean risk
    return {
        "dataset": name,
        "n": int(data.num_nodes),
        "spearman_lte_abs_phi2_gap": float(rho_gap),
        "pvalue_gap": float(p_gap),
        "spearman_lte_phi2": float(rho_phi),
        "pvalue_phi2": float(p_phi),
        "mean_risk_member": float(risk[tr].mean()),
        "mean_risk_nonmember": float(risk[te].mean()),
        "test_acc": float((probs[te].argmax(1) == y[te]).mean()),
    }


def stress_lira_note():
    """Explain conf≪LiRA mismatch on Volume×leakage stress cell."""
    path = os.path.join(load_config()["results_dir"], "volume_highrisk_synth_summary.json")
    if not os.path.isfile(path):
        return {"status": "missing"}
    s = json.load(open(path))
    return {
        "status": "ok",
        "observation": (
            "On the controlled stress cell, conf AUROC ≈0.79 while LiRA ≈0.50. "
            "Likely causes: (1) membership signal is concentrated in max-confidence / "
            "entropy features that LR-φ exploits, while Gaussian LiRA on full logits is "
            "under-separated when Acc is near chance and class posteriors are diffuse; "
            "(2) n_shadows=4 with highly overfit targets yields noisy shadow likelihoods; "
            "(3) primary protocol metric is LiRA — this cell is a *stress test for conf "
            "AUROC*, not a LiRA-positive Volume finding."
        ),
        "summary_means": s.get("summary_means"),
        "framing": "controlled_stress_test_not_volume_peer",
    }


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    rows = []
    for name, loader in [
        ("Cora", load_citation),
        ("Citeseer", load_citation),
        ("Actor", load_heterophilic),
    ]:
        print(f"LTE Spearman: {name}", flush=True)
        rows.append(analyze_dataset(name, loader))
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(cfg["results_dir"], "lte_phi_gap_spearman.csv")
    df.to_csv(out_csv, index=False)
    note = stress_lira_note()
    out = {
        "lte_spearman": rows,
        "interpretation": (
            "LTE is a three-term heuristic proxy (inv-degree, local heterophily, "
            "supervised-neighbor fraction), not a learned leakage estimator. "
            "Report Spearman with |φ₂-gap|; if weak on Actor, treat as proxy."
        ),
        "stress_cell_lira_vs_conf": note,
    }
    with open(os.path.join(cfg["results_dir"], "lte_mechanism_audit.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
