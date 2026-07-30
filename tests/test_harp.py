"""Unit tests for HARP hop expansion and scale allocation."""
import numpy as np
import torch
from torch_geometric.data import Data

from defenses.harp import expand_k_hop, select_risk_seeds, compute_harp_scales


def test_expand_k_hop_line():
    # 0-1-2-3 path
    ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    seed = np.array([True, False, False, False])
    assert expand_k_hop(ei, seed, 0).tolist() == [True, False, False, False]
    assert expand_k_hop(ei, seed, 1).tolist() == [True, True, False, False]
    assert expand_k_hop(ei, seed, 2).tolist() == [True, True, True, False]
    assert expand_k_hop(ei, seed, 3).tolist() == [True, True, True, True]


def test_select_top_frac():
    r = np.array([0.1, 0.9, 0.5, 0.8])
    m = select_risk_seeds(r, risk_frac=0.5)
    assert m.sum() == 2
    assert m[1] and m[3]


def test_target_protect_frac():
    n = 100
    # star-like: hub 0 connected to all
    src = [0] * (n - 1) + list(range(1, n))
    dst = list(range(1, n)) + [0] * (n - 1)
    ei = torch.tensor([src, dst], dtype=torch.long)
    data = Data(edge_index=ei, num_nodes=n, y=torch.zeros(n, dtype=torch.long),
                train_mask=torch.zeros(n, dtype=torch.bool))
    data.train_mask[:20] = True
    data.x = torch.randn(n, 4)
    risk = torch.linspace(0, 1, n)
    scales, prot, seeds, stats = compute_harp_scales(
        data, risk=risk, k_hops=1, strong_noise_scale=0.3,
        target_protect_frac=0.4,
    )
    assert abs(stats["frac_protected"] - 0.4) < 0.08
    assert scales[prot].max() == 0.3
    assert (scales[~prot] == 0).all()


if __name__ == "__main__":
    test_expand_k_hop_line()
    test_select_top_frac()
    test_target_protect_frac()
    print("ok")
