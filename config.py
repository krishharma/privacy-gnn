"""
Configuration and paths for the PrivacyGNN project.
Loads experiment grid from experiment_config.yaml when available (config-driven).
"""
import hashlib
import os
import yaml

# Project root (directory containing this file)
ROOT = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")
FIGURES_DIR = os.path.join(ROOT, "figures")

DEVICE = "cpu"
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

DEFAULT_ATTACKS = ["confidence", "threshold", "shadow", "lira"]

DATASETS = [
    "Cora",
    "Citeseer",
    "PubMed",
    "synthetic_high_sparse",
    "synthetic_high_medium",
    "synthetic_high_dense",
    "synthetic_low_sparse",
    "synthetic_low_medium",
    "synthetic_low_dense",
]
MODELS = ["LogReg", "MLP", "GCN", "GraphSAGE", "GAT"]
DEFENSES = [
    ("none", {}),
    ("dropedge", {"rate": 0.3}),
    ("label_smoothing", {"alpha": 0.1}),
    ("early_stopping", {"patience": 15}),
    ("confidence_masking", {"top_k": 2}),
    ("edge_sparsification", {"rate": 0.2}),
    ("lbp", {"scale": 0.3}),
    ("gtd", {"gamma": 1.0, "stage1_frac": 0.5, "pseudo_conf": 0.8}),
    ("sami", {"lam": 0.1, "use_lte": True, "use_gate": True, "beta": 0.0}),
    ("harp", {
        "lam": 0.5, "use_lte": True, "use_gate": True, "arch_aware": True,
        "risk_frac": 0.30, "k_hops": 1, "strong_noise_scale": 0.30,
        "weak_noise_scale": 0.0, "target_protect_frac": 0.40,
        "warmup_epochs": 5, "entropy_coef": 0.05,
        "train_on_protected": True,
    }),
    # Ablations (named defenses for the confirmatory ablation table)
    ("sami_no_lte", {"lam": 0.1, "use_lte": False, "use_gate": True, "beta": 0.0}),
    ("sami_no_adv", {"lam": 0.0, "use_lte": True, "use_gate": True, "beta": 0.0}),
    ("sami_no_gate", {"lam": 0.1, "use_lte": True, "use_gate": False, "beta": 0.0}),
    ("sami_temp_only", {"lam": 0.0, "use_lte": True, "use_gate": False, "beta": 1.0}),
    ("advreg", {"lam": 0.1, "use_lte": False, "use_gate": False, "beta": 0.0}),
    ("harp_k0", {
        "lam": 0.5, "use_lte": True, "use_gate": True, "risk_frac": 0.40,
        "k_hops": 0, "strong_noise_scale": 0.30, "weak_noise_scale": 0.0,
        "target_protect_frac": 0.40, "warmup_epochs": 5, "train_on_protected": True,
    }),
    ("harp_k2", {
        "lam": 0.5, "use_lte": True, "use_gate": True, "risk_frac": 0.20,
        "k_hops": 2, "strong_noise_scale": 0.30, "weak_noise_scale": 0.0,
        "target_protect_frac": 0.40, "warmup_epochs": 5, "train_on_protected": True,
    }),
    ("harp_uniform", {
        "lam": 0.5, "use_lte": True, "use_gate": False, "risk_frac": 1.0,
        "k_hops": 0, "strong_noise_scale": 0.30, "weak_noise_scale": 0.0,
        "target_protect_frac": None, "warmup_epochs": 5, "train_on_protected": False,
    }),
    ("harp_release_only", {
        "lam": 0.0, "use_lte": True, "use_gate": False, "risk_frac": 0.30,
        "k_hops": 1, "strong_noise_scale": 0.30, "weak_noise_scale": 0.0,
        "target_protect_frac": 0.40, "warmup_epochs": 5, "train_on_protected": False,
    }),
]

# Feature-only models only get the no-defense cell (plus LBP post-hoc if listed).
_FEATURE_ONLY = ("LogReg", "MLP")
_GNN_ONLY_DEFENSES = {
    "dropedge",
    "label_smoothing",
    "early_stopping",
    "edge_sparsification",
    "gtd",
    "sami",
    "sami_no_lte",
    "sami_no_adv",
    "sami_no_gate",
    "sami_temp_only",
    "advreg",
    "harp",
    "harp_k0",
    "harp_k2",
    "harp_uniform",
    "harp_release_only",
}


def load_config(config_path=None):
    """Load YAML if it exists; otherwise use defaults. Honors PRIVACYGNN_CONFIG when path is None."""
    if config_path is None:
        cfg_name = os.environ.get("PRIVACYGNN_CONFIG", "experiment_config.yaml")
        config_path = cfg_name if os.path.isabs(cfg_name) else os.path.join(ROOT, cfg_name)
    if not os.path.isfile(config_path):
        return _default_config_dict()
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    raw_def = raw.get("defenses")
    if raw_def is None:
        defenses = DEFENSES
    else:
        defenses = [(d["name"], d.get("params", {})) for d in raw_def]
    split = raw.get("split", {"train_ratio": 0.4, "val_ratio": 0.2, "test_ratio": 0.4})
    cfg = {
        "seeds": raw.get("seeds", SEEDS),
        "datasets": raw.get("datasets", DATASETS),
        "models": raw.get("models", MODELS),
        "defenses": defenses,
        "results_dir": os.path.join(ROOT, raw.get("results_dir", "results")),
        "data_dir": os.path.join(ROOT, raw.get("data_dir", "data")),
        "figures_dir": os.path.join(ROOT, raw.get("figures_dir", "figures")),
        "training": raw.get("training", {}),
        "attacks": [a.lower() for a in raw.get("attacks", DEFAULT_ATTACKS)],
        "lira": raw.get("lira", {"n_shadows": 8}),
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
        "dp_sgd_datasets": raw.get("dp_sgd_datasets"),
        "split": split,
        "stats": raw.get("stats", {"multiple_comparison": "holm"}),
        "_config_path": os.path.abspath(config_path),
    }
    # Config hash for reproducibility metadata in result rows.
    try:
        with open(config_path, "rb") as bf:
            cfg["config_hash"] = hashlib.sha256(bf.read()).hexdigest()[:12]
    except OSError:
        cfg["config_hash"] = "unknown"
    return cfg


def _default_config_dict():
    return {
        "_config_path": None,
        "config_hash": "defaults",
        "seeds": SEEDS,
        "datasets": DATASETS,
        "models": MODELS,
        "defenses": DEFENSES,
        "results_dir": RESULTS_DIR,
        "data_dir": DATA_DIR,
        "figures_dir": FIGURES_DIR,
        "training": {},
        "attacks": list(DEFAULT_ATTACKS),
        "lira": {"n_shadows": 8},
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
        "split": {"train_ratio": 0.4, "val_ratio": 0.2, "test_ratio": 0.4},
        "stats": {"multiple_comparison": "holm"},
    }


def get_experiment_list(config=None):
    """Build list of (dataset, model, defense_name, defense_params, seed)."""
    if config is None:
        config = load_config()
    exps = []
    for ds in config["datasets"]:
        for model in config["models"]:
            for dn, dp in config["defenses"]:
                if model in _FEATURE_ONLY and dn in _GNN_ONLY_DEFENSES:
                    continue
                if model in _FEATURE_ONLY and dn not in ("none", "confidence_masking", "lbp"):
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
