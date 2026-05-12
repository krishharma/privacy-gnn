"""
Configuration and paths for the PrivacyGNN project.
Loads experiment grid from experiment_config.yaml when available (config-driven).
"""
import os
import yaml

# Project root (directory containing this file)
ROOT = os.path.dirname(os.path.abspath(__file__))

# Default paths
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")
FIGURES_DIR = os.path.join(ROOT, "figures")

# Device and seeds (overridden by config file if present)
DEVICE = "cpu"
SEEDS = [42, 123, 456, 789, 1024]

DEFAULT_ATTACKS = ["confidence", "threshold", "shadow", "lira"]

# Default experiment grid (overridden by config file if present)
DATASETS = [
    "Cora",
    "Citeseer",
    "synthetic_high_sparse",
    "synthetic_high_medium",
    "synthetic_high_dense",
    "synthetic_low_sparse",
    "synthetic_low_medium",
    "synthetic_low_dense",
]
MODELS = ["LogReg", "MLP", "GCN", "GraphSAGE"]
DEFENSES = [
    ("none", {}),
    ("dropedge", {"rate": 0.3}),
    ("label_smoothing", {"alpha": 0.1}),
    ("early_stopping", {"patience": 15}),
    ("confidence_masking", {"top_k": 2}),
    ("edge_sparsification", {"rate": 0.2}),
]


def load_config(config_path=None):
    """Load YAML if it exists; otherwise use defaults. Honors PRIVACYGNN_CONFIG when path is None."""
    if config_path is None:
        cfg_name = os.environ.get("PRIVACYGNN_CONFIG", "experiment_config.yaml")
        config_path = cfg_name if os.path.isabs(cfg_name) else os.path.join(ROOT, cfg_name)
    if not os.path.isfile(config_path):
        return _default_config_dict()
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    # Convert defenses from list of {name, params} to list of (name, params)
    raw_def = raw.get("defenses")
    if raw_def is None:
        defenses = DEFENSES
    else:
        defenses = [(d["name"], d.get("params", {})) for d in raw_def]
    return {
        "seeds": raw.get("seeds", SEEDS),
        "datasets": raw.get("datasets", DATASETS),
        "models": raw.get("models", MODELS),
        "defenses": defenses,
        "results_dir": os.path.join(ROOT, raw.get("results_dir", "results")),
        "data_dir": os.path.join(ROOT, raw.get("data_dir", "data")),
        "figures_dir": os.path.join(ROOT, raw.get("figures_dir", "figures")),
        "training": raw.get("training", {}),
        "attacks": [a.lower() for a in raw.get("attacks", DEFAULT_ATTACKS)],
        "lira": raw.get("lira", {"n_shadows": 3}),
        "bootstrap": raw.get("bootstrap", {"n_resamples": 1000, "confidence": 0.95}),
        "minibatch": raw.get("minibatch", {"batch_size": 1024, "num_neighbors": [15, 10]}),
        "device": raw.get("device", DEVICE),
        "large_graph_use_official_split": raw.get("large_graph_use_official_split", True),
        "dp_sgd": raw.get(
            "dp_sgd",
            {
                "max_grad_norm": 1.0,
                "noise_multiplier": 1.0,
                "delta": 1e-5,
                "epochs": 20,
                "batch_size": 1024,
                "lr": 0.05,
            },
        ),
        # If set, only run defense dp_sgd on these datasets (keeps runtime manageable).
        "dp_sgd_datasets": raw.get("dp_sgd_datasets"),
        "_config_path": os.path.abspath(config_path),
    }


def _default_config_dict():
    return {
        "_config_path": None,
        "seeds": SEEDS,
        "datasets": DATASETS,
        "models": MODELS,
        "defenses": DEFENSES,
        "results_dir": RESULTS_DIR,
        "data_dir": DATA_DIR,
        "figures_dir": FIGURES_DIR,
        "training": {},
        "attacks": list(DEFAULT_ATTACKS),
        "lira": {"n_shadows": 3},
        "bootstrap": {"n_resamples": 1000, "confidence": 0.95},
        "minibatch": {"batch_size": 1024, "num_neighbors": [15, 10]},
        "device": DEVICE,
        "large_graph_use_official_split": True,
        "dp_sgd": {
            "max_grad_norm": 1.0,
            "noise_multiplier": 1.0,
            "delta": 1e-5,
            "epochs": 20,
            "batch_size": 1024,
            "lr": 0.05,
        },
        "dp_sgd_datasets": None,
    }


def get_experiment_list(config=None):
    """Build list of (dataset, model, defense_name, defense_params, seed)."""
    if config is None:
        config = load_config()
    exps = []
    for ds in config["datasets"]:
        for model in config["models"]:
            for dn, dp in config["defenses"]:
                if model in ("LogReg", "MLP") and dn != "none":
                    continue
                if dn == "dp_sgd" and model not in ("GCN", "GraphSAGE"):
                    continue
                if dn == "dp_sgd":
                    allow = config.get("dp_sgd_datasets")
                    if allow is not None and ds not in allow:
                        continue
                for seed in config["seeds"]:
                    exps.append((ds, model, dn, dp, seed))
    return exps


def ensure_dirs(config=None):
    """Create results, data, and figures directories."""
    if config is None:
        config = load_config()
    for d in (config["results_dir"], config["data_dir"], config["figures_dir"]):
        os.makedirs(d, exist_ok=True)
