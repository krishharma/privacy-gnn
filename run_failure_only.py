#!/usr/bin/env python3
"""Run Chameleon qualitative failure-case analysis only."""
import torch
from run_harp_rejection_fix import cfg, run_failure_cases

if __name__ == "__main__":
    device = torch.device("cpu")
    c = cfg()
    print("START failure", flush=True)
    run_failure_cases(device, c)
    print("DONE failure", flush=True)
