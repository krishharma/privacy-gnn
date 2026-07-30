"""
Large-graph loaders (OGB / Reddit / LINKX arxiv-year) for Volume-scale SPAB audits.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

MINIBATCH_DATASETS = frozenset({"ogbn-arxiv", "arxiv-year", "Reddit", "ogbn-products"})


def _masks_from_idx(n: int, train_idx, val_idx, test_idx):
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _even_quantile_labels(vals: np.ndarray, nclasses: int = 5) -> np.ndarray:
    """LINKX even-quantile binning (Lim et al., Non-Homophilous Large Scale)."""
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        inds = (vals >= lower) & (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    return label


def _linkx_random_split_masks(n: int, seed: int = 0, train_prop: float = 0.5, valid_prop: float = 0.25):
    """Deterministic LINKX-style 50/25/25 split (arxiv-year has no OGB official split)."""
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n)
    n_train = int(n * train_prop)
    n_val = int(n * valid_prop)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return _masks_from_idx(n, train_idx, val_idx, test_idx)


def _patch_torch_load_for_pyg():
    """OGB pickled Data under torch>=2.6 needs weights_only=False."""
    if getattr(torch.load, "_privacygnn_patched", False):
        return
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._privacygnn_patched = True
    torch.load = _load


def load_large_benchmark(name: str, root: str):
    """
    Load a large node-classification graph with train/val/test masks.
    Returns (Data, num_classes, num_features).
    """
    name = str(name)
    os.makedirs(root, exist_ok=True)
    _patch_torch_load_for_pyg()

    if name == "ogbn-arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset

        ds = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
        data = ds[0]
        # OGB stores directed citation edges; undirected helps SAGE/GCN.
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        y = data.y
        if y.dim() > 1:
            y = y.view(-1)
        data.y = y.long()
        split = ds.get_idx_split()
        n = data.num_nodes
        data.train_mask, data.val_mask, data.test_mask = _masks_from_idx(
            n, split["train"], split["valid"], split["test"]
        )
        num_classes = int(ds.num_classes)
        num_features = int(data.x.size(1))
        return data, num_classes, num_features

    if name == "arxiv-year":
        # Same topology/features as ogbn-arxiv; year-quantile labels → heterophilic peer
        # (Lim et al., LINKX / Non-Homophilous Large-Scale).
        from ogb.nodeproppred import PygNodePropPredDataset

        ds = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
        data = ds[0]
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        years = data.node_year.view(-1).detach().cpu().numpy().astype(np.float64)
        y = _even_quantile_labels(years, nclasses=5)
        data.y = torch.as_tensor(y, dtype=torch.long)
        n = int(data.num_nodes)
        # Canonical fixed LINKX-style split (no OGB official year split exists).
        data.train_mask, data.val_mask, data.test_mask = _linkx_random_split_masks(n, seed=0)
        num_classes = 5
        num_features = int(data.x.size(1))
        return data, num_classes, num_features

    if name == "ogbn-products":
        from ogb.nodeproppred import PygNodePropPredDataset

        ds = PygNodePropPredDataset(name="ogbn-products", root=root)
        data = ds[0]
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        y = data.y.view(-1).long() if data.y.dim() > 1 else data.y.long()
        data.y = y
        split = ds.get_idx_split()
        n = data.num_nodes
        data.train_mask, data.val_mask, data.test_mask = _masks_from_idx(
            n, split["train"], split["valid"], split["test"]
        )
        return data, int(ds.num_classes), int(data.x.size(1))

    if name == "Reddit":
        from torch_geometric.datasets import Reddit

        ds = Reddit(root=root)
        data = ds[0]
        return data, int(ds.num_classes), int(ds.num_node_features)

    raise ValueError(f"Unknown large benchmark: {name}")
