"""
ogbn-products systems comparison without full LiRA (16GB host).

Trains GraphSAGE briefly for none / strong LBP / locked HARP release.
Reports subsample Acc, Mass, Frac, setup/release wall time, peak RSS.
"""
from __future__ import annotations

import json
import os
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import ensure_dirs, load_config
from defenses.harp import LOCKED_HARP, compute_harp_scales
from defenses.lbp import lbp_perturb
from graph_minibatch import train_gnn_minibatch
from models import SAGE
from ogb_loader import load_large_benchmark
from sklearn.metrics import accuracy_score

OUT = "results/ogbn_products_harp_systems.json"
N_EVAL = 50000


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def subsample_acc(probs, y, mask, n_eval=N_EVAL, seed=0):
    idx = np.where(mask)[0]
    rng = np.random.default_rng(seed)
    if len(idx) > n_eval:
        idx = rng.choice(idx, size=n_eval, replace=False)
    pred = probs[idx].argmax(1)
    return float(accuracy_score(y[idx], pred)), int(len(idx))


def main():
    os.environ["PRIVACYGNN_CONFIG"] = "experiment_config_ogbn.yaml"
    cfg = dict(load_config())
    ensure_dirs(cfg)
    root = os.path.join(cfg["data_dir"], "ogb")
    device = torch.device("cpu")

    import builtins
    import ogb.utils.url as ogb_url

    ogb_url.decide_download = lambda url: True
    _in = builtins.input
    builtins.input = lambda *a, **k: "y"
    try:
        print("Loading ogbn-products...", flush=True)
        data, nc, nf = load_large_benchmark("ogbn-products", root)
    finally:
        builtins.input = _in

    n = int(data.num_nodes)
    meta = {
        "n_nodes": n,
        "n_edges": int(data.edge_index.size(1)),
        "num_classes": int(nc),
        "peak_rss_mb_after_load": round(peak_rss_mb(), 1),
    }
    print(meta, flush=True)

    mb = cfg.get("minibatch", {})
    batch_size = min(int(mb.get("batch_size", 1024)), 256)
    num_neighbors = [10, 5]
    epochs = 5

    torch.manual_seed(42)
    model = SAGE(nf, 64, nc)
    t0 = time.time()
    train_gnn_minibatch(
        model, data, device, data.edge_index,
        epochs=epochs, lr=0.01, weight_decay=5e-4,
        batch_size=batch_size, num_neighbors=num_neighbors,
    )
    train_s = time.time() - t0
    model.eval()
    # NeighborLoader full softmax is heavy; use a light forward on a random eval set
    # via stored train-time pattern: sample nodes and run SAGE on their neighborhoods
    # through a single full-batch is impossible; reuse systems_only style eval.
    from torch_geometric.loader import NeighborLoader

    te_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    if te_idx.numel() > N_EVAL:
        perm = torch.randperm(te_idx.numel())[:N_EVAL]
        te_idx = te_idx[perm]
    loader = NeighborLoader(
        data, num_neighbors=num_neighbors, batch_size=512,
        input_nodes=te_idx, shuffle=False,
    )
    probs = np.zeros((n, nc), dtype=np.float32)
    filled = np.zeros(n, dtype=bool)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = F.softmax(model(batch.x, batch.edge_index)[: batch.batch_size], dim=-1)
            root = batch.n_id[: batch.batch_size].cpu().numpy()
            probs[root] = out.cpu().numpy()
            filled[root] = True
    y = data.y.view(-1).cpu().numpy()
    eval_mask = filled & data.test_mask.cpu().numpy()

    # HARP scales (LTE) — setup cost
    t1 = time.time()
    scales, prot, _, hstats = compute_harp_scales(
        data.cpu(),
        risk=None,
        risk_frac=LOCKED_HARP["risk_frac"],
        k_hops=LOCKED_HARP["k_hops"],
        strong_noise_scale=LOCKED_HARP["strong_noise_scale"],
        weak_noise_scale=0.0,
        target_protect_frac=LOCKED_HARP["target_protect_frac"],
        arch="sage",
        arch_aware=True,
    )
    harp_setup_s = time.time() - t1
    mass_harp = float(np.asarray(scales).sum())
    frac = float(hstats["frac_protected"])

    # Release transforms on eval nodes only (memory-safe)
    eval_idx = np.where(eval_mask)[0]
    p_eval = probs[eval_idx]
    y_eval = y[eval_idx]
    scales_e = np.asarray(scales)[eval_idx]

    t2 = time.time()
    p_none = p_eval
    acc_none = float(accuracy_score(y_eval, p_none.argmax(1)))
    none_rel_s = time.time() - t2

    t2 = time.time()
    p_lbp = lbp_perturb(p_eval, scale=0.3, seed=42)
    acc_lbp = float(accuracy_score(y_eval, p_lbp.argmax(1)))
    lbp_rel_s = time.time() - t2
    mass_lbp = 0.3 * n

    t2 = time.time()
    # HARP: noise only where scale>0
    noise = np.random.default_rng(42).laplace(0.0, np.maximum(scales_e, 1e-12)[:, None], size=p_eval.shape)
    noise = np.where(scales_e[:, None] > 0, noise, 0.0)
    p_harp = np.clip(p_eval + noise, 0, None)
    p_harp = p_harp / np.maximum(p_harp.sum(1, keepdims=True), 1e-12)
    acc_harp = float(accuracy_score(y_eval, p_harp.argmax(1)))
    harp_rel_s = time.time() - t2

    out = {
        **meta,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_neighbors": num_neighbors,
        "n_eval": int(eval_mask.sum()),
        "train_seconds": round(train_s, 2),
        "harp_setup_seconds": round(harp_setup_s, 2),
        "none": {"acc": acc_none, "mass": None, "frac": 0.0, "release_s": round(none_rel_s, 4)},
        "lbp_b0.3": {"acc": acc_lbp, "mass": mass_lbp, "frac": 1.0, "release_s": round(lbp_rel_s, 4)},
        "harp": {
            "acc": acc_harp,
            "mass": mass_harp,
            "frac": frac,
            "release_s": round(harp_rel_s, 4),
            "setup_s": round(harp_setup_s, 2),
        },
        "delta_acc_harp_minus_lbp": round(acc_harp - acc_lbp, 4),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "note": "Subsample NeighborLoader Acc; no LiRA (OOM on 16GB). Mass/Frac from full-graph HARP scales.",
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
