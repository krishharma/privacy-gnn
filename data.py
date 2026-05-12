"""
Datasets and graph utilities for PrivacyGNN.
- Citation networks: Cora, Citeseer (via PyTorch Geometric Planetoid).
- Synthetic graphs with controlled homophily and density.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


def get_data_dir():
    """Data directory from config (lazy import to avoid circular import)."""
    from config import load_config
    return load_config()["data_dir"]


# Mirror for Planetoid when default GitHub raw fails (e.g. network/DNS). Same path layout as original.
_PLANETOID_MIRROR = "https://gitee.com/jiajiewu/planetoid/raw/master/data"

def load_citation(name, data_dir=None):
    """Load Cora or Citeseer via Planetoid. Tries default URL, then PyG mirror. Returns (data, num_classes, num_features)."""
    if data_dir is None:
        data_dir = get_data_dir()
    try:
        ds = Planetoid(root=data_dir, name=name)
        return ds[0], ds.num_classes, ds.num_node_features
    except Exception:
        # Fallback: PyG mirror (no trailing slash to match Planetoid's path joining)
        orig_url = getattr(Planetoid, "url", None)
        try:
            Planetoid.url = _PLANETOID_MIRROR
            ds = Planetoid(root=data_dir, name=name)
            return ds[0], ds.num_classes, ds.num_node_features
        finally:
            if orig_url is not None:
                Planetoid.url = orig_url


def make_synthetic(n=400, nf=50, nc=5, homo="high", dens="medium", seed=42):
    """
    Generate a synthetic graph with controlled homophily and density.
    Returns (Data, num_classes, num_features).
    """
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, nc, n)
    centers = rng.randn(nc, nf) * 2
    feats = np.array([centers[l] + rng.randn(nf) * 0.8 for l in labels])

    density_map = {"sparse": 0.005, "medium": 0.015, "dense": 0.04}
    target_edges = int(density_map[dens] * n * (n - 1) / 2)
    homophily_frac = {"low": 0.3, "high": 0.8}[homo]
    n_same = int(target_edges * homophily_frac)
    n_diff = target_edges - n_same

    src, dst, edge_set = [], [], set()

    # Same-label edges
    for _ in range(n_same * 10):
        if len(src) // 2 >= n_same:
            break
        i = rng.randint(0, n)
        same = np.where(labels == labels[i])[0]
        if len(same) < 2:
            continue
        j = same[rng.randint(0, len(same))]
        if i != j and (i, j) not in edge_set:
            edge_set.add((i, j))
            edge_set.add((j, i))
            src.extend([i, j])
            dst.extend([j, i])

    # Different-label edges
    added = 0
    for _ in range(n_diff * 10):
        if added >= n_diff:
            break
        i = rng.randint(0, n)
        diff = np.where(labels != labels[i])[0]
        if len(diff) < 1:
            continue
        j = diff[rng.randint(0, len(diff))]
        if i != j and (i, j) not in edge_set:
            edge_set.add((i, j))
            edge_set.add((j, i))
            src.extend([i, j])
            dst.extend([j, i])
            added += 1

    if len(src) == 0:
        src, dst = [0, 1], [1, 0]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.tensor(feats, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    perm = rng.permutation(n)
    n_train = n // 2
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    test_mask[perm[n_train:]] = True

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask), nc, nf


def resplit(data, seed):
    """Create a new random train/test split (50/50) with the given seed."""
    rng = np.random.RandomState(seed)
    n = data.num_nodes
    perm = rng.permutation(n)
    n_train = n // 2
    d = data.clone()
    d.train_mask = torch.zeros(n, dtype=torch.bool)
    d.test_mask = torch.zeros(n, dtype=torch.bool)
    d.train_mask[perm[:n_train]] = True
    d.test_mask[perm[n_train:]] = True
    # Clear val_mask if present (official OGB/Reddit splits) so training uses only train_mask.
    if hasattr(d, "val_mask") and d.val_mask is not None:
        d.val_mask = torch.zeros(n, dtype=torch.bool)
    return d


def homophily(data):
    """Fraction of edges that connect nodes with the same label."""
    if data.edge_index.size(1) == 0:
        return 0.0
    s, t = data.edge_index
    return (data.y[s] == data.y[t]).float().mean().item()


def density(data):
    """Undirected edge density (edges / possible edges)."""
    n = data.num_nodes
    m = data.edge_index.size(1) // 2
    return m / (n * (n - 1) / 2) if n > 1 else 0.0


def drop_edges(edge_index, rate):
    """Randomly drop edges with probability `rate`. Returns subset of edge_index."""
    ne = edge_index.size(1)
    keep = torch.rand(ne) > rate
    if keep.any():
        return edge_index[:, keep]
    return edge_index


def drop_edges_undirected(edge_index, rate):
    """
    Randomly drop undirected edge pairs with probability `rate`.

    PyG stores undirected graphs as two directed columns, (u, v) and (v, u).
    Edge sparsification should keep or remove both directions together so the
    defense corresponds to thinning the underlying undirected graph.
    """
    if rate <= 0 or edge_index.size(1) == 0:
        return edge_index

    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    pair_to_cols = {}
    for col, (u, v) in enumerate(zip(src, dst)):
        key = (u, v) if u <= v else (v, u)
        pair_to_cols.setdefault(key, []).append(col)

    kept_cols = []
    for cols in pair_to_cols.values():
        if torch.rand(()) > rate:
            kept_cols.extend(cols)

    if not kept_cols:
        return edge_index

    kept_cols.sort()
    keep = torch.tensor(kept_cols, dtype=torch.long, device=edge_index.device)
    return edge_index[:, keep]
