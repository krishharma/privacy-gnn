import yaml
import time
from experiment import run_one

with open('experiment_config_ogbn.yaml') as f:
    config = yaml.safe_load(f)

t0 = time.time()
try:
    res = run_one(
        "ogbn-arxiv", "GCN", "none",
        config["defenses"][0]["params"],
        42,
        config=config,
        device="cpu"
    )
    print(f"Done in {time.time()-t0:.1f}s:", res["test_accuracy"])
except BaseException as e:
    import traceback
    traceback.print_exc()
