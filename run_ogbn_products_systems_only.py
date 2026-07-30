"""
ogbn-products systems-only probe (16GB host OOMs on full Acc+MIA).
Train GraphSAGE none briefly; report Acc + wall + peak RSS. No conf/LiRA.
"""
from __future__ import annotations

import json
import os
import resource
import time
import traceback

import torch

from config import ensure_dirs, load_config
from graph_minibatch import train_gnn_minibatch, measure_api_qps
from models import SAGE
from ogb_loader import load_large_benchmark
from sklearn.metrics import accuracy_score

OUT = "results/ogbn_products_systems_only.json"
PROBE = "results/ogbn_products_probe.json"


def peak_rss_mb():
    # macOS: ru_maxrss is bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


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

    meta = {
        "n_nodes": int(data.num_nodes),
        "n_edges": int(data.edge_index.size(1)),
        "num_classes": int(nc),
        "num_features": int(nf),
        "train_n": int(data.train_mask.sum()),
        "val_n": int(data.val_mask.sum()),
        "test_n": int(data.test_mask.sum()),
        "rss_mb_after_load": round(peak_rss_mb(), 1),
    }
    print(meta, flush=True)

    mb = cfg.get("minibatch", {})
    batch_size = min(int(mb.get("batch_size", 1024)), 256)
    num_neighbors = [10, 5]
    epochs = 5

    torch.manual_seed(42)
    model = SAGE(nf, 64, nc)
    t0 = time.time()
    try:
        train_gnn_minibatch(
            model,
            data,
            device,
            data.edge_index,
            epochs=epochs,
            lr=0.01,
            weight_decay=5e-4,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
        )
    except Exception as e:
        out = {
            "status": "train_failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            **meta,
        }
        json.dump(out, open(OUT, "w"), indent=2)
        print(out, flush=True)
        raise

    train_s = round(time.time() - t0, 2)
    model.eval()
    # Eval Acc on a subsample of test to avoid materializing 2.2M logits
    te = data.test_mask.nonzero(as_tuple=False).view(-1)
    n_eval = min(50000, int(te.numel()))
    idx = te[torch.randperm(te.numel())[:n_eval]]
    with torch.no_grad():
        # full-batch infer may OOM — use NeighborLoader-style if available
        try:
            from torch_geometric.loader import NeighborLoader

            loader = NeighborLoader(
                data,
                num_neighbors=[-1, -1],
                batch_size=1024,
                input_nodes=idx,
                shuffle=False,
            )
            preds, ys = [], []
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                # seed nodes are first batch.batch_size in NeighborLoader
                bs = batch.batch_size
                preds.append(out[:bs].argmax(1).cpu())
                ys.append(batch.y[:bs].cpu())
            pred = torch.cat(preds).numpy()
            y = torch.cat(ys).numpy()
            acc = float(accuracy_score(y, pred))
            eval_mode = "neighborloader_subsample"
        except Exception as e:
            acc = float("nan")
            eval_mode = f"eval_failed:{e}"

    qps = None
    try:
        qps = measure_api_qps(
            model, data, device, num_neighbors=[10, 5], batch_size=256
        )
    except Exception:
        pass

    out = {
        "status": "systems_only_ok",
        "dataset": "ogbn-products",
        "defense": "none",
        "epochs": epochs,
        "batch_size": batch_size,
        "num_neighbors": num_neighbors,
        "train_seconds": train_s,
        "test_accuracy_subsample": round(acc, 4) if acc == acc else None,
        "n_eval": n_eval,
        "eval_mode": eval_mode,
        "api_qps_approx": qps,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "note": "Full MIA (conf/LiRA) OOMs on 16GB host; systems+subsample Acc only.",
        **meta,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    probe = {
        "attempted": "ogbn-products",
        "status": "systems_only_complete",
        "n_nodes": meta["n_nodes"],
        "full_mia": "OOM exit 137 (n_shadows=1 and n_shadows=2)",
        "systems_only": OUT,
        "summary": out,
    }
    json.dump(probe, open(PROBE, "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
