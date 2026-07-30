"""
Documented Volume×Variety probes for SPAB (exploratory → freeze if leaky).

Finding (2026-07): official ogbn stays near-chance under resplit/small-train;
feature-noise collapses Acc; n=10k low-SNR GraphSAGE Acc≈1 with conf≈chance.
Primary multi-thousand-node high-risk cell = Actor (see baselines_extra.csv /
actor_summary.json). This script records that negative result for Veracity.
"""
from __future__ import annotations

import json
import os

from config import ensure_dirs, load_config


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    out = {
        "purpose": "Volume×Variety probe log for IEEE BigData SPAB",
        "ogbn_official": "Volume negative control; conf/LiRA near chance",
        "ogbn_resplit_40_20_40": "Still near-chance (conf≈0.50); scale≠leakage",
        "ogbn_featnoise": "Acc collapses (~0.06); not a useful leaky Volume cell",
        "synth_n10k_low_sparse_snr0.25_sage": "Acc≈1.0, conf≈0.50–0.53; φ-gap closed",
        "primary_high_risk_multik": {
            "dataset": "Actor",
            "n_nodes": "~7600",
            "artifact": "results/actor_summary.json",
            "note": "Heterophilic Variety at multi-thousand-node scale; 5 seeds",
        },
        "recommendation": (
            "Paper: ogbn = Volume negative control + systems; Actor = Variety high-risk; "
            "do not force a fake leaky ogbn resplit."
        ),
    }
    path = os.path.join(cfg["results_dir"], "volume_variety_probe.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
