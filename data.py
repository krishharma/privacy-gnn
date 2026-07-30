"""
Datasets and graph utilities for PrivacyGNN.
- Citation networks: Cora, Citeseer (via PyTorch Geometric Planetoid).
- Heterophilic graphs: Actor, Chameleon, Squirrel.
- Synthetic graphs with controlled homophily and density.
"""
import os
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
    """Load Cora, Citeseer, or PubMed via Planetoid. Tries default URL, then mirror."""
    if data_dir is None:
        data_dir = get_data_dir()
    try:
        ds = Planetoid(root=data_dir, name=name)
        return ds[0], ds.num_classes, ds.num_node_features
    except Exception:
        orig_url = getattr(Planetoid, "url", None)
        try:
            Planetoid.url = _PLANETOID_MIRROR
            ds = Planetoid(root=data_dir, name=name)
            return ds[0], ds.num_classes, ds.num_node_features
        finally:
            if orig_url is not None:
                Planetoid.url = orig_url


def load_heterophilic(name, data_dir=None):
    """
    Load real heterophilic graphs: Actor, Chameleon, or Squirrel (PyG).
    Used to stress-test structure-dependent leakage beyond citation networks.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    name = str(name)
    if name == "Actor":
        from torch_geometric.datasets import Actor

        ds = Actor(root=os.path.join(data_dir, "Actor"))
        return ds[0], ds.num_classes, ds.num_node_features
    if name in ("Chameleon", "Squirrel"):
        from torch_geometric.datasets import WikipediaNetwork

        ds = WikipediaNetwork(root=os.path.join(data_dir, "WikipediaNetwork"), name=name.lower())
        return ds[0], ds.num_classes, ds.num_node_features
    raise ValueError(f"Unknown heterophilic dataset: {name}")

def apply_split_masks_counts(data, seed, n_train, n_val, n_test):
    """
    Assign train/val/test masks with fixed counts (Planetoid-style).
    Remaining nodes are unlabeled. Members for MIA = train_mask; non-members = test_mask.
    """
    rng = np.random.RandomState(seed)
    n = int(data.num_nodes)
    n_train = max(1, min(int(n_train), n - 2))
    n_val = max(1, min(int(n_val), n - n_train - 1))
    n_test = max(1, min(int(n_test), n - n_train - n_val))
    perm = rng.permutation(n)
    d = data.clone() if hasattr(data, "clone") else data
    d.train_mask = torch.zeros(n, dtype=torch.bool)
    d.val_mask = torch.zeros(n, dtype=torch.bool)
    d.test_mask = torch.zeros(n, dtype=torch.bool)
    d.train_mask[perm[:n_train]] = True
    d.val_mask[perm[n_train : n_train + n_val]] = True
    d.test_mask[perm[n_train + n_val : n_train + n_val + n_test]] = True
    return d


def apply_split_masks(data, seed, train_ratio=0.4, val_ratio=0.2, test_ratio=0.4):
    """
    Assign train/val/test masks with the given ratios (default 40/20/40).
    Ratios are renormalized if they do not sum to 1. Members for MIA = train_mask;
    non-members = test_mask; val is used only for early stopping / defense proxies.
    """
    total = float(train_ratio + val_ratio + test_ratio)
    if total <= 0:
        train_ratio, val_ratio, test_ratio = 0.4, 0.2, 0.4
        total = 1.0
    train_ratio, val_ratio, test_ratio = (
        train_ratio / total,
        val_ratio / total,
        test_ratio / total,
    )
    rng = np.random.RandomState(seed)
    n = data.num_nodes
    perm = rng.permutation(n)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Remainder goes to test so masks partition V.
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    d = data.clone() if hasattr(data, "clone") else data
    d.train_mask = torch.zeros(n, dtype=torch.bool)
    d.val_mask = torch.zeros(n, dtype=torch.bool)
    d.test_mask = torch.zeros(n, dtype=torch.bool)
    d.train_mask[perm[:n_train]] = True
    d.val_mask[perm[n_train : n_train + n_val]] = True
    d.test_mask[perm[n_train + n_val :]] = True
    return d


def make_synthetic(
    n=400,
    nf=50,
    nc=5,
    homo="high",
    dens="medium",
    seed=42,
    train_ratio=0.4,
    val_ratio=0.2,
    test_ratio=0.4,
    feature_snr=1.0,
    h_frac=None,
    density_value=None,
):
    """
    Generate a synthetic graph with controlled homophily, density, and feature SNR.

    feature_snr: class-mean separation / noise scale (default 1.0 ≈ centers*2 / noise 0.8).
      Low SNR → structure-dominated leakage; high SNR → feature-dominated (MLP reverse).
    Optional h_frac / density_value override named homo/dens maps for denser SCML grids.
    """
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, nc, n)
    snr = float(feature_snr)
    # Default historical: centers * 2, noise * 0.8 → relative SNR ≈ 2/0.8 = 2.5
    center_scale = 2.0 * snr
    noise_scale = 0.8
    centers = rng.randn(nc, nf) * center_scale
    feats = np.array([centers[l] + rng.randn(nf) * noise_scale for l in labels])

    density_map = {
        "sparse": 0.005,
        "medium": 0.015,
        "dense": 0.04,
        "xsparse": 0.0025,
        "xdense": 0.08,
    }
    if density_value is not None:
        dens_v = float(density_value)
    else:
        dens_v = density_map.get(dens, density_map["medium"])
    target_edges = int(dens_v * n * (n - 1) / 2)
    if h_frac is not None:
        homophily_frac = float(h_frac)
    else:
        homophily_frac = {"low": 0.3, "mid": 0.5, "high": 0.8}.get(homo, 0.8)
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
    data = Data(x=x, edge_index=edge_index, y=y)
    data = apply_split_masks(data, seed, train_ratio, val_ratio, test_ratio)
    data.feature_snr = float(snr)
    data.target_homophily = float(homophily_frac)
    data.target_density = float(dens_v)
    return data, nc, nf


def resplit(data, seed, train_ratio=0.4, val_ratio=0.2, test_ratio=0.4):
    """Create a new random train/val/test split (default 40/20/40) with the given seed."""
    return apply_split_masks(data, seed, train_ratio, val_ratio, test_ratio)


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
