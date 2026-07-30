#!/usr/bin/env python3
"""Run products subsample LiRA only (uses existing ogb data under data/ogb/)."""
import torch
from run_harp_rejection_fix import run_products_sub_lira

if __name__ == "__main__":
    device = torch.device("cpu")
    print("START products-sub", flush=True)
    run_products_sub_lira(device, n_sub=15000)
    print("DONE products-sub", flush=True)
