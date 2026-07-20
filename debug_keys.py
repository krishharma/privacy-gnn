from config import load_config
from experiment import run_one
config = load_config("experiment_config_paper.yaml")
res = run_one("Cora", "GCN", "none", {}, 42, config=config)
print(list(res.keys()))
