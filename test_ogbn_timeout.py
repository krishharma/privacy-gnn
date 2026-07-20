import yaml
import time
import signal
from experiment import run_one

def handler(signum, frame):
    raise Exception("TIMEOUT!")
signal.signal(signal.SIGALRM, handler)
signal.alarm(10)

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
except Exception as e:
    import traceback
    traceback.print_exc()
