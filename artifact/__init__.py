"""
Defense implementations for PrivacyGNN.

- harp: Hop-Aware Risk-conditioned Privacy (primary contribution).
- sami: Structure-Aware Membership Indistinguishability (supporting / prior).
- lbp:  Laplacian binned posterior perturbation (Olatunji et al., reimplementation).
- gtd:  Graph Transductive Defense style train-test alternate + flattening (reimplementation).
"""
from defenses.sami import (
    compute_lte_risk,
    phi_features_torch,
    train_gnn_sami,
    risk_scaled_temperature,
    risk_scaled_posterior_noise,
)
from defenses.harp import (
    compute_harp_scales,
    expand_k_hop,
    select_risk_seeds,
    LOCKED_HARP,
)
from defenses.lbp import lbp_perturb
from defenses.gtd import train_gnn_gtd

__all__ = [
    "compute_lte_risk",
    "phi_features_torch",
    "train_gnn_sami",
    "risk_scaled_temperature",
    "risk_scaled_posterior_noise",
    "compute_harp_scales",
    "expand_k_hop",
    "select_risk_seeds",
    "LOCKED_HARP",
    "lbp_perturb",
    "train_gnn_gtd",
]
