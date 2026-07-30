"""
LBP: Laplacian binned posterior perturbation (reimplementation).

Following Olatunji, Nejdl, and Khosla, "Membership Inference Attack on Graph
Neural Networks" (2021): posterior entries are grouped into bins by confidence
rank and a single Laplace noise draw is added per bin (all entries in the same
bin receive the same noise), which preserves more utility than i.i.d. noise on
every entry (VanPd). Perturbed posteriors are clipped at zero and renormalized.

Applied post hoc to released posteriors only; the trained model is unchanged.
The same transform is applied to shadow-model posteriors so the attacker faces
the identical API.
"""
import numpy as np


def lbp_perturb(p, scale=0.3, n_bins=None, seed=0):
    """
    Perturb posteriors p [n, C] with rank-binned Laplace noise.

    - scale: Laplace scale b (larger = more noise = stronger defense).
    - n_bins: number of rank bins; defaults to C (one bin per confidence rank).
    - Same noise draw per (row, bin), assigned by descending-confidence rank,
      so the relative ordering signal an attacker exploits is disrupted while
      per-class calibration degrades gracefully.
    """
    p = np.array(p, dtype=float, copy=True)
    n, c = p.shape
    if n_bins is None or n_bins <= 0:
        n_bins = c
    n_bins = min(n_bins, c)
    rng = np.random.RandomState(int(seed))

    # Rank classes per row (0 = most confident).
    order = np.argsort(-p, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(c)[None, :]
    # Map ranks to bins.
    bin_of_rank = (np.arange(c) * n_bins) // c
    bins = bin_of_rank[ranks]

    noise_per_bin = rng.laplace(0.0, scale, size=(n, n_bins))
    p = p + noise_per_bin[rows.repeat(c, axis=1), bins]

    p = np.clip(p, 0.0, None)
    row_sum = p.sum(axis=1, keepdims=True)
    # Rows zeroed by noise fall back to uniform.
    uniform = np.full((1, c), 1.0 / c)
    p = np.where(row_sum > 0, p / np.clip(row_sum, 1e-12, None), uniform)
    return p
