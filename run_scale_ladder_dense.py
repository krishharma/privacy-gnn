"""
Denser scale ladder + large-graph shadow stability.

1) Append Chameleon (from chameleon_baselines.csv) to the matched-budget ladder.
2) Controlled ogbn-arxiv node-subsamples at 5k/10k/20k/50k (n_shadows=4, 3 seeds).
3) PubMed shadow sweep n_shadows in {2,4,8} (3 seeds) to stress shadow-fidelity
   as a confounder of the scale null.

Writes:
  results/scale_ladder_dense.csv / _means.csv / _summary.json
  results/pubmed_shadow_sweep.csv / _summary.json
  figures/fig_scale_ladder.png  (regenerated)
"""
from __future__ import annotations

import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from config import ensure_dirs, load_config
from data import density, homophily, resplit
from experiment import _resplit_kwargs, _split_kwargs, run_one
from lira_attack import lira_gaussian_auc
from models import SAGE
from training import train_gnn

SEEDS = [42, 123, 456]
N_SHADOWS = 4
OG_SIZES = [5000, 10000, 20000, 50000]
PUBMED_SHADOWS = [2, 4, 8]


def _patch_torch_load():
    _orig = torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)

    torch.load = _load


def _run_local(name, data, nf, nc, device, seed, n_shadows=N_SHADOWS, epochs=50):
    model = SAGE(nf, 64, nc)
    train_gnn(model, data, device, epochs=epochs)
    model.eval()
    with torch.no_grad():
        p = F.softmax(model(data.x.to(device), data.edge_index.to(device)).cpu(), 1).numpy()
    y = data.y.numpy()
    tr, te = data.train_mask.numpy(), data.test_mask.numpy()
    acc = float(accuracy_score(y[te], p[te].argmax(1)))
    shadow_p, shadow_tr, shadow_te = [], [], []
    t0 = time.time()
    for k in range(n_shadows):
        sh_seed = seed + 999 + k * 10007
        sh = resplit(data.clone(), sh_seed)
        m = SAGE(nf, 64, nc)
        train_gnn(m, sh, device, epochs=epochs)
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
        "n_shadows": n_shadows,
        "acc": acc,
        "lira": float(lira),
        "tpr01": float(tpr),
        "train_s": time.time() - t0,
        "defense": "none",
        "family": "subsample" if name.startswith("ogbn-") else "off_shelf",
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
        y=data.y[keep].clone().view(-1),
    )
    return resplit(out, seed)


def _chameleon_rows(cfg):
    path = os.path.join(cfg["results_dir"], "chameleon_baselines.csv")
    df = pd.read_csv(path)
    g = df[(df.model == "GraphSAGE") & (df.defense.astype(str).str.lower() == "none")]
    rows = []
    for _, r in g.iterrows():
        rows.append(
            {
                "dataset": "Chameleon",
                "n_nodes": 2277,
                "homophily": float(r.get("homophily", 0.235)),
                "density": float(r.get("density", 0.006966)),
                "seed": int(r["seed"]),
                "n_shadows": 4,
                "acc": float(r["test_accuracy"]),
                "lira": float(r["lira_attack_auc"]),
                "tpr01": float(r.get("lira_tpr_at_0.01_fpr", float("nan"))),
                "train_s": float(r.get("train_seconds", float("nan"))),
                "defense": "none",
                "family": "off_shelf",
                "hetero": True,
            }
        )
    return rows


def _plot(means, out_png):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    m = means.sort_values("n_nodes")
    colors = []
    for _, r in m.iterrows():
        if r["dataset"] in ("Actor", "Chameleon"):
            colors.append("#c45c26")
        elif str(r["dataset"]).startswith("ogbn-") and r["dataset"] != "ogbn-arxiv":
            colors.append("#2a6f97")
        else:
            colors.append("#1b4332")
    ax.scatter(m["n_nodes"], m["lira"], c=colors, s=55, zorder=3)
    for _, r in m.iterrows():
        ax.annotate(
            r["dataset"],
            (r["n_nodes"], r["lira"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.axhline(0.5, color="0.5", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Nodes (log)")
    ax.set_ylabel("Undefended LiRA AUROC")
    ax.set_ylim(0.45, 0.66)
    ax.set_title("Matched-budget score LiRA vs scale ($n_{shadows}=4$)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)


def main():
    _patch_torch_load()
    cfg = dict(load_config())
    ensure_dirs(cfg)
    device = torch.device(cfg.get("device", "cpu"))
    ratios = _resplit_kwargs(_split_kwargs(cfg))
    rows = []

    # Prior ladder (off-the-shelf)
    prior = os.path.join(cfg["results_dir"], "scale_ladder.csv")
    if os.path.isfile(prior):
        old = pd.read_csv(prior)
        old["family"] = old["dataset"].apply(
            lambda d: "subsample" if str(d).startswith("ogbn-") and d != "ogbn-arxiv" else "off_shelf"
        )
        old["hetero"] = old["dataset"].isin(["Actor", "Chameleon"])
        rows.extend(old.to_dict(orient="records"))

    # Chameleon from existing baselines
    rows.extend(_chameleon_rows(cfg))

    # Controlled ogbn-arxiv subsamples
    try:
        from ogb.nodeproppred import PygNodePropPredDataset

        try:
            import ogb.utils.util as ou

            ou.decide_download = lambda *a, **k: True
        except Exception:
            pass
        ds = PygNodePropPredDataset(name="ogbn-arxiv", root=os.path.join(cfg["data_dir"], "ogbn"))
        og = ds[0]
        if og.y.dim() > 1:
            og.y = og.y.view(-1)
        nc = int(og.y.max()) + 1
        nf = int(og.x.size(1))
        for n_keep in OG_SIZES:
            for seed in SEEDS:
                name = f"ogbn-{n_keep // 1000}k"
                print(f"ladder {name} seed={seed}", flush=True)
                data = _subsample(og, n_keep, seed)
                row = _run_local(name, data, nf, nc, device, seed, epochs=50)
                row["hetero"] = False
                rows.append(row)
                print(row, flush=True)
    except Exception as e:
        print("ogbn subsample failed", e, flush=True)

    df = pd.DataFrame(rows)
    # Drop duplicate (dataset, seed, n_shadows) preferring newest
    df = df.drop_duplicates(subset=["dataset", "seed", "n_shadows"], keep="last")
    out_csv = os.path.join(cfg["results_dir"], "scale_ladder_dense.csv")
    df.to_csv(out_csv, index=False)
    means = (
        df[df.n_shadows == 4]
        .groupby("dataset", as_index=False)[["n_nodes", "acc", "lira"]]
        .mean()
        .sort_values("n_nodes")
    )
    means.to_csv(os.path.join(cfg["results_dir"], "scale_ladder_dense_means.csv"), index=False)
    print(means.to_string())

    # PubMed shadow sweep (fidelity check on a near-chance large graph)
    sweep = []
    c = dict(cfg)
    c["attacks"] = ["lira", "confidence"]
    for ns in PUBMED_SHADOWS:
        c["lira"] = {"n_shadows": ns}
        for seed in SEEDS:
            print(f"pubmed shadows={ns} seed={seed}", flush=True)
            r = run_one("PubMed", "GraphSAGE", "none", {}, seed, config=c)
            sweep.append(
                {
                    "dataset": "PubMed",
                    "n_nodes": 19717,
                    "seed": seed,
                    "n_shadows": ns,
                    "acc": r["test_accuracy"],
                    "lira": r["lira_attack_auc"],
                    "defense": "none",
                }
            )
            print(sweep[-1], flush=True)
    sw = pd.DataFrame(sweep)
    sw.to_csv(os.path.join(cfg["results_dir"], "pubmed_shadow_sweep.csv"), index=False)
    sw_means = sw.groupby("n_shadows", as_index=False)[["acc", "lira"]].mean()
    sw_means.to_csv(os.path.join(cfg["results_dir"], "pubmed_shadow_sweep_means.csv"), index=False)
    print(sw_means.to_string())

    summary = {
        "means": means.to_dict(orient="records"),
        "n_shadows": N_SHADOWS,
        "pubmed_shadow_sweep": sw_means.to_dict(orient="records"),
        "ogbn_sizes": OG_SIZES,
    }
    with open(os.path.join(cfg["results_dir"], "scale_ladder_dense_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    _plot(means, os.path.join(fig_dir, "fig_scale_ladder.png"))
    # also paper_visuals
    pv = os.path.join(os.path.dirname(__file__), "paper", "paper_visuals")
    os.makedirs(pv, exist_ok=True)
    _plot(means, os.path.join(pv, "fig_scale_ladder.png"))


if __name__ == "__main__":
    main()
