import torch
import yaml
import time
from experiment import run_one

with open('experiment_config_paper.yaml') as f:
    config = yaml.safe_load(f)

print("Starting Cora/GCN/dp_sgd...")
try:
    t0 = time.time()
    res = run_one(
        "Cora", "GCN", "dp_sgd", 
        config["defenses"][-1]["params"], 
        42,
        config=config,
        device="cpu"
    )
    print(f"Done in {time.time()-t0:.1f}s:", res["test_accuracy"])
except Exception as e:
    import traceback
    traceback.print_exc()
