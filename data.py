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


def make_synthetic(n=400, nf=50, nc=5, homo="high", dens="medium", seed=42, center_std=2.0, noise_std=0.8):
    """
    Generate a synthetic graph with controlled homophily and density.
    Returns (Data, num_classes, num_features). Data has a .stats dict.
    """
    import hashlib
    import networkx as nx
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, nc, n)
    centers = rng.randn(nc, nf) * center_std
    feats = np.array([centers[l] + rng.randn(nf) * noise_std for l in labels])

    density_map = {"sparse": 0.005, "medium": 0.015, "dense": 0.04}
    req_density = density_map[dens]
    target_edges = int(req_density * n * (n - 1) / 2)
    homophily_frac = {"low": 0.3, "medium": 0.55, "high": 0.8}.get(homo, 0.8)
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
    n_train = int(n * 0.4)
    n_val = int(n * 0.1)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train+n_val]] = True
    test_mask[perm[n_train+n_val:]] = True

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    # Compute stats
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = [(src[i], dst[i]) for i in range(len(src))]
    G.add_edges_from(edges)
    
    degrees = [d for _, d in G.degree()]
    components = list(nx.connected_components(G))
    
    # feature separation
    try:
        clf = LogisticRegression(max_iter=200, random_state=seed)
        cv_score = float(cross_val_score(clf, feats, labels, cv=3).mean())
    except Exception:
        cv_score = float('nan')
        
    s_t, t_t = data.edge_index
    realized_homo = float((data.y[s_t] == data.y[t_t]).float().mean().item()) if s_t.numel() > 0 else 0.0
    realized_dens = float((data.edge_index.size(1) / 2) / (n * (n - 1) / 2)) if n > 1 else 0.0
    
    # Hashing
    graph_bytes = feats.tobytes() + labels.tobytes() + edge_index.numpy().tobytes() + train_mask.numpy().tobytes() + val_mask.numpy().tobytes()
    ghash = hashlib.sha256(graph_bytes).hexdigest()

    data.stats = {
        "req_density": float(req_density),
        "realized_density": realized_dens,
        "req_homophily": float(homophily_frac),
        "realized_homophily": realized_homo,
        "nodes": int(n),
        "edges": int(G.number_of_edges()),
        "components": int(len(components)),
        "largest_comp_frac": float(len(max(components, key=len)) / n) if components else 0.0,
        "isolated_frac": float(sum(1 for d in degrees if d == 0) / n),
        "degree_mean": float(np.mean(degrees)),
        "degree_median": float(np.median(degrees)),
        "degree_max": float(np.max(degrees)),
        "degree_var": float(np.var(degrees)),
        "class_counts": np.bincount(labels, minlength=nc).tolist(),
        "train_class_counts": np.bincount(labels[train_mask.numpy()], minlength=nc).tolist(),
        "val_class_counts": np.bincount(labels[val_mask.numpy()], minlength=nc).tolist(),
        "test_class_counts": np.bincount(labels[test_mask.numpy()], minlength=nc).tolist(),
        "feat_cv_score": cv_score,
        "center_std": float(center_std),
        "noise_std": float(noise_std),
        "seed": int(seed),
        "graph_hash": ghash
    }
    
    return data, nc, nf


def resplit(data, seed):
    """Create a new random train/val/test split (40/10/50) with the given seed."""
    rng = np.random.RandomState(seed)
    n = data.num_nodes
    perm = rng.permutation(n)
    n_train = int(n * 0.4)
    n_val = int(n * 0.1)
    d = data.clone()
    d.train_mask = torch.zeros(n, dtype=torch.bool)
    d.val_mask = torch.zeros(n, dtype=torch.bool)
    d.test_mask = torch.zeros(n, dtype=torch.bool)
    d.train_mask[perm[:n_train]] = True
    d.val_mask[perm[n_train:n_train+n_val]] = True
    d.test_mask[perm[n_train+n_val:]] = True
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
