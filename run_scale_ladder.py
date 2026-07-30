"""
Matched-budget leakage-vs-scale ladder (none GraphSAGE, n_shadows=4).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import Amazon

from config import ensure_dirs, load_config
from data import resplit, homophily, density
from experiment import run_one, _resplit_kwargs, _split_kwargs
from lira_attack import lira_gaussian_auc
from models import SAGE
from training import train_gnn

N_SHADOWS = 4
SEEDS = [42, 123, 456]


def _run_local(name, data, nf, nc, device, seed):
    model = SAGE(nf, 64, nc)
    train_gnn(model, data, device, epochs=50)
    model.eval()
    with torch.no_grad():
        p = F.softmax(model(data.x.to(device), data.edge_index.to(device)).cpu(), 1).numpy()
    y = data.y.numpy()
    tr, te = data.train_mask.numpy(), data.test_mask.numpy()
    acc = float(accuracy_score(y[te], p[te].argmax(1)))
    shadow_p, shadow_tr, shadow_te = [], [], []
    t0 = time.time()
    for k in range(N_SHADOWS):
        sh_seed = seed + 999 + k * 10007
        sh = resplit(data.clone(), sh_seed)
        m = SAGE(nf, 64, nc)
        train_gnn(m, sh, device, epochs=50)
        m.eval()
        with torch.no_grad():
            sp = F.softmax(m(sh.x.to(device), sh.edge_index.to(device)).cpu(), 1).numpy()
        shadow_p.append(sp)
        shadow_tr.append(sh.train_mask.numpy())
        shadow_te.append(sh.test_mask.numpy())
    lira, _, _, tpr = lira_gaussian_auc(p, y, tr, te, shadow_p, shadow_tr, shadow_te)
    return {
        "dataset": name,
        "n_nodes": int(data.num_nodes),
        "homophily": float(homophily(data)),
        "density": float(density(data)),
        "seed": seed,
        "n_shadows": N_SHADOWS,
        "acc": acc,
        "lira": float(lira),
        "tpr01": float(tpr),
        "train_s": time.time() - t0,
        "defense": "none",
    }


def _subsample(data, n_keep, seed):
    rng = np.random.default_rng(seed)
    n = int(data.num_nodes)
    keep = np.sort(rng.choice(n, size=min(n_keep, n), replace=False))
    mapping = -np.ones(n, dtype=np.int64)
    mapping[keep] = np.arange(len(keep))
    ei = data.edge_index.cpu().numpy()
    m = (mapping[ei[0]] >= 0) & (mapping[ei[1]] >= 0)
    ei2 = np.stack([mapping[ei[0][m]], mapping[ei[1][m]]], 0)
    from torch_geometric.data import Data
    out = Data(
        x=data.x[keep].clone(),
        edge_index=torch.from_numpy(ei2).long(),
        y=data.y[keep].clone(),
    )
    return resplit(out, seed)


def main():
    cfg = dict(load_config())
    ensure_dirs(cfg)
    device = torch.device(cfg.get("device", "cpu"))
    ratios = _resplit_kwargs(_split_kwargs(cfg))
    rows = []
    c = dict(cfg)
    c["lira"] = {"n_shadows": N_SHADOWS}
    c["attacks"] = ["lira", "confidence"]

    sizes = {"Cora": 2708, "Citeseer": 3327, "PubMed": 19717, "Actor": 7600}
    for ds in ["Cora", "Citeseer", "PubMed", "Actor"]:
        for seed in SEEDS:
            print(f"ladder {ds} seed={seed}", flush=True)
            r = run_one(ds, "GraphSAGE", "none", {}, seed, config=c)
            rows.append({
                "dataset": ds,
                "n_nodes": sizes[ds],
                "homophily": r.get("homophily", float("nan")),
                "density": r.get("density", float("nan")),
                "seed": seed,
                "n_shadows": N_SHADOWS,
                "acc": r["test_accuracy"],
                "lira": r["lira_attack_auc"],
                "tpr01": r.get("lira_tpr_at_0.01_fpr", float("nan")),
                "train_s": r.get("train_seconds", float("nan")),
                "defense": "none",
            })
            print(rows[-1], flush=True)

    for amazon_name in ["Computers", "Photo"]:
        try:
            root = os.path.join(cfg["data_dir"], amazon_name)
            g = Amazon(root=root, name=amazon_name)[0]
            nc = int(g.y.max()) + 1
            nf = int(g.x.size(1))
            for seed in SEEDS:
                print(f"ladder {amazon_name} seed={seed}", flush=True)
                data = resplit(g.clone(), seed, **ratios)
                rows.append(_run_local(amazon_name, data, nf, nc, device, seed))
                print(rows[-1], flush=True)
        except Exception as e:
            print(amazon_name, "failed", e, flush=True)

    # Prefer canonical Computers Acc/LiRA from run_one grid (computers_baselines.csv).
    # The lean Amazon-local path underfits Acc (~0.57) while LiRA stays ~0.50.
    cb_path = os.path.join(cfg["results_dir"], "computers_baselines.csv")
    if os.path.isfile(cb_path):
        cb = pd.read_csv(cb_path)
        g = cb[(cb.model == "GraphSAGE") & (cb.defense.astype(str).str.lower() == "none")]
        fixed = []
        for r in rows:
            if r.get("dataset") == "Computers":
                match = g[g.seed == r["seed"]]
                if len(match):
                    m = match.iloc[0]
                    r = dict(r)
                    r["acc"] = float(m.test_accuracy)
                    r["lira"] = float(m.lira_attack_auc)
                    if "lira_tpr_at_0.01_fpr" in m.index:
                        r["tpr01"] = float(m["lira_tpr_at_0.01_fpr"])
            fixed.append(r)
        rows = fixed
        print("Computers Acc synced from computers_baselines.csv", flush=True)

    # ogbn-arxiv subsampled
    try:
        from ogb.nodeproppred import PygNodePropPredDataset

        def _yes(*a, **k):
            return True

        try:
            from ogb.utils.util import decide_download as _dd
            import ogb.utils.util as ou
            ou.decide_download = _yes
        except Exception:
            pass
        ds = PygNodePropPredDataset(name="ogbn-arxiv", root=os.path.join(cfg["data_dir"], "ogbn"))
        og = ds[0]
        if hasattr(og, "y") and og.y.dim() > 1:
            og.y = og.y.view(-1)
        nc = int(og.y.max()) + 1
        nf = int(og.x.size(1))
        for n_keep in [20000, 50000]:
            for seed in SEEDS:
                name = f"ogbn-{n_keep // 1000}k"
                print(f"ladder {name} seed={seed}", flush=True)
                data = _subsample(og, n_keep, seed)
                rows.append(_run_local(name, data, nf, nc, device, seed))
                print(rows[-1], flush=True)
    except Exception as e:
        print("ogbn subsample failed", e, flush=True)

    # full ogbn n=4 from prior CSV
    p = os.path.join(cfg["results_dir"], "ogbn_lira_n4_3seed.csv")
    if os.path.isfile(p):
        ogbn = pd.read_csv(p)
        col_acc = "test_accuracy" if "test_accuracy" in ogbn.columns else "acc"
        col_lira = "lira_attack_auc" if "lira_attack_auc" in ogbn.columns else "lira"
        for _, r in ogbn[ogbn.defense.astype(str).str.lower() == "none"].iterrows():
            rows.append({
                "dataset": "ogbn-arxiv",
                "n_nodes": 169343,
                "homophily": float("nan"),
                "density": float("nan"),
                "seed": int(r["seed"]) if "seed" in r else 0,
                "n_shadows": 4,
                "acc": float(r[col_acc]),
                "lira": float(r[col_lira]),
                "tpr01": float("nan"),
                "train_s": float("nan"),
                "defense": "none",
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg["results_dir"], "scale_ladder.csv"), index=False)
    means = (
        df.groupby("dataset", as_index=False)[["n_nodes", "acc", "lira"]]
        .mean()
        .sort_values("n_nodes")
    )
    means.to_csv(os.path.join(cfg["results_dir"], "scale_ladder_means.csv"), index=False)
    print(means.to_string())
    with open(os.path.join(cfg["results_dir"], "scale_ladder_summary.json"), "w") as f:
        json.dump({"means": means.to_dict(orient="records"), "n_shadows": N_SHADOWS}, f, indent=2)


if __name__ == "__main__":
    main()
