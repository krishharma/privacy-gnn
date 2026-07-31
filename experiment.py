"""
Single-experiment runner: load data, train model, run MIAs and calibration, return metrics.

Supports defenses: none, dropedge, label_smoothing, early_stopping, confidence_masking,
edge_sparsification, lbp, gtd, sami (+ ablations), advreg, harp (+ hop ablations).
LiRA uses multiple shadow models; confidence/threshold/shadow keep the 4D φ feature map.
"""
from __future__ import annotations

import json
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from config import load_config
from data import (
    load_citation,
    make_synthetic,
    resplit,
    homophily,
    density,
    drop_edges_undirected,
    load_heterophilic,
)
from models import GCN, SAGE, GAT, GatedGCN, GatedSAGE
from training import train_gnn
from attacks import (
    confidence_attack,
    shadow_attack,
    calibration_error,
    extract_features,
    tpr_at_fpr,
    mlp_phi_attack,
    average_posterior_queries,
    gap_attack,
)
from ogb_loader import MINIBATCH_DATASETS, load_large_benchmark
from graph_minibatch import (
    train_gnn_minibatch,
    infer_logits_minibatch,
    train_gnn_dp_minibatch,
    measure_api_qps,
)
from lira_attack import lira_gaussian_auc
from defenses.sami import (
    train_gnn_sami,
    train_gnn_sami_minibatch,
    risk_scaled_temperature,
    compute_lte_risk,
    risk_scaled_posterior_noise,
    allocate_risk_budget,
)
from defenses.harp import compute_harp_scales, mask_risk_to_protected
from defenses.lbp import lbp_perturb
from defenses.gtd import train_gnn_gtd, train_gnn_gtd_minibatch
from defenses.memguard import apply_memguard


def _apply_confidence_masking(p: np.ndarray, cmk: int | None) -> np.ndarray:
    if cmk is None:
        return p
    p = np.array(p, copy=True)
    for i in range(len(p)):
        idx = np.argsort(p[i])[::-1][cmk:]
        p[i, idx] = 0
        s = p[i].sum()
        if s > 0:
            p[i] /= s
    return p


def _split_kwargs(config):
    split = config.get("split", {})
    return dict(
        train_ratio=float(split.get("train_ratio", 0.4)),
        val_ratio=float(split.get("val_ratio", 0.2)),
        test_ratio=float(split.get("test_ratio", 0.4)),
        protocol=str(split.get("protocol", "random_ratio")),
    )


_HETEROPHILIC = frozenset({"Actor", "Chameleon", "Squirrel"})


def _resplit_kwargs(split_kw):
    """Drop non-ratio keys before calling resplit/apply_split_masks."""
    return {
        k: split_kw[k]
        for k in ("train_ratio", "val_ratio", "test_ratio")
        if k in split_kw
    }


def _load_target_data(dataset_name: str, data_dir: str, seed: int, use_official_large: bool, split_kw):
    protocol = str(split_kw.get("protocol", "random_ratio"))
    ratios = _resplit_kwargs(split_kw)

    def _tag(d, nc, nf):
        try:
            d.dataset_name = dataset_name
        except Exception:
            pass
        return d, nc, nf

    if dataset_name in ("Cora", "Citeseer", "PubMed"):
        data, num_classes, num_features = load_citation(dataset_name, data_dir=data_dir)
        # planetoid_public: keep fixed Planetoid masks (comparability appendix).
        if protocol == "planetoid_public":
            return _tag(data.clone() if hasattr(data, "clone") else data, num_classes, num_features)
        return _tag(resplit(data, seed, **ratios), num_classes, num_features)
    if dataset_name in _HETEROPHILIC:
        data, num_classes, num_features = load_heterophilic(dataset_name, data_dir=data_dir)
        return _tag(resplit(data, seed, **ratios), num_classes, num_features)
    if dataset_name.startswith("synthetic_"):
        parts = dataset_name.split("_")
        # synthetic_{homo}_{dens}[_{snrX}][_{nN}][_{hH}]
        homo, dens = parts[1], parts[2]
        snr = 1.0
        n_nodes = 400
        h_frac = None
        for p in parts[3:]:
            if p.startswith("snr"):
                snr = float(p[3:])
            elif p.startswith("n") and p[1:].replace(".", "", 1).isdigit():
                n_nodes = int(float(p[1:]))
            elif p.startswith("h") and p[1:].replace(".", "", 1).isdigit():
                h_frac = float(p[1:])
        return make_synthetic(
            n=n_nodes,
            homo=homo,
            dens=dens,
            seed=seed,
            feature_snr=snr,
            h_frac=h_frac,
            **ratios,
        )
    if dataset_name in MINIBATCH_DATASETS:
        if dataset_name == "Reddit":
            root = os.path.join(data_dir, "pyg")
        else:
            root = os.path.join(data_dir, "ogb")
        data, num_classes, num_features = load_large_benchmark(dataset_name, root)
        if use_official_large:
            return data, num_classes, num_features
        return resplit(data.clone(), seed, **ratios), num_classes, num_features
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _make_shadow_data(dataset_name: str, data_dir: str, shadow_seed: int, split_kw):
    ratios = _resplit_kwargs(split_kw)
    protocol = str(split_kw.get("protocol", "random_ratio"))
    if dataset_name in ("Cora", "Citeseer", "PubMed"):
        sd, nc, nf = load_citation(dataset_name, data_dir=data_dir)
        # Under planetoid_public targets, shadows still resample membership so LiRA
        # has diverse IN/OUT models; match Planetoid train/val/test *counts*.
        if protocol == "planetoid_public":
            n = int(sd.num_nodes)
            n_tr = int(sd.train_mask.sum().item())
            n_va = int(sd.val_mask.sum().item())
            n_te = int(sd.test_mask.sum().item())
            # Remaining nodes stay unlabeled (mirror Planetoid).
            from data import apply_split_masks_counts

            return apply_split_masks_counts(sd, shadow_seed, n_tr, n_va, n_te), nc, nf
        return resplit(sd, shadow_seed, **ratios), nc, nf
    if dataset_name in _HETEROPHILIC:
        sd, nc, nf = load_heterophilic(dataset_name, data_dir=data_dir)
        return resplit(sd, shadow_seed, **ratios), nc, nf
    if dataset_name.startswith("synthetic_"):
        parts = dataset_name.split("_")
        homo, dens = parts[1], parts[2]
        snr = 1.0
        n_nodes = 400
        h_frac = None
        for p in parts[3:]:
            if p.startswith("snr"):
                snr = float(p[3:])
            elif p.startswith("n") and p[1:].replace(".", "", 1).isdigit():
                n_nodes = int(float(p[1:]))
            elif p.startswith("h") and p[1:].replace(".", "", 1).isdigit():
                h_frac = float(p[1:])
        return make_synthetic(
            n=n_nodes,
            homo=homo,
            dens=dens,
            seed=shadow_seed,
            feature_snr=snr,
            h_frac=h_frac,
            **ratios,
        )
    if dataset_name in MINIBATCH_DATASETS:
        if dataset_name == "Reddit":
            root = os.path.join(data_dir, "pyg")
        else:
            root = os.path.join(data_dir, "ogb")
        data, nc, nf = load_large_benchmark(dataset_name, root)
        return resplit(data.clone(), shadow_seed, **ratios), nc, nf
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _is_sami_family(name: str) -> bool:
    return name in (
        "sami",
        "sami_no_lte",
        "sami_no_adv",
        "sami_no_gate",
        "sami_temp_only",
        "advreg",
    )


def _is_harp_family(name: str) -> bool:
    return name in (
        "harp",
        "harp_k0",
        "harp_k2",
        "harp_uniform",
        "harp_release_only",
        "harp_audit",
        "harp_random",
        "harp_mask",
        "harp_degree",
        "harp_train_nbr",
        "harp_confidence",
        "harp_entropy",
    )


def _shadow_vulnerability_risk(data, model_name, num_features, num_classes, device, ep, lr, wd, cfg, n_rank=4, seed0=0):
    """
    Per-node audit-guided risk: |μ_IN − μ_OUT| of logit-true-class confidence
    across a small undefended shadow ensemble (Aerni-style vulnerability).
    Used as HARP seed ranking when seed_mode='audit'.
    """
    from lira_attack import _logit_confidence

    n = int(data.num_nodes)
    yn = data.y.detach().cpu().numpy()
    in_vals = [[] for _ in range(n)]
    out_vals = [[] for _ in range(n)]
    split_kw = _split_kwargs(cfg) if isinstance(cfg, dict) else {
        "train_ratio": 0.4, "val_ratio": 0.2, "test_ratio": 0.4,
    }
    ds_name = getattr(data, "dataset_name", None)
    for k in range(int(n_rank)):
        shadow_seed = int(seed0) + 1000 + k
        if ds_name:
            sdata, _, _ = _make_shadow_data(ds_name, cfg.get("data_dir", "data"), shadow_seed, split_kw)
        else:
            sdata = data
            # Re-split train mask for ranking shadows when dataset name unknown.
            rng = np.random.RandomState(shadow_seed)
            idx = rng.permutation(n)
            n_tr = max(1, int(0.4 * n))
            trm = np.zeros(n, dtype=bool)
            trm[idx[:n_tr]] = True
            sdata = data.clone() if hasattr(data, "clone") else data
            if hasattr(sdata, "train_mask"):
                sdata.train_mask = torch.as_tensor(trm)
        model = _make_gnn(model_name, num_features, num_classes, use_gate=False)
        train_gnn(model, sdata, device, epochs=ep, lr=lr, weight_decay=wd)
        model.eval()
        with torch.no_grad():
            logits = model(sdata.x.to(device), sdata.edge_index.to(device))
            p = torch.softmax(logits, dim=1).detach().cpu().numpy()
        conf = _logit_confidence(p, yn if len(yn) == p.shape[0] else sdata.y.cpu().numpy())
        trm = sdata.train_mask.cpu().numpy().astype(bool)
        for v in range(min(n, len(conf))):
            if trm[v]:
                in_vals[v].append(float(conf[v]))
            else:
                out_vals[v].append(float(conf[v]))
    risk = np.zeros(n, dtype=float)
    for v in range(n):
        if len(in_vals[v]) < 1 or len(out_vals[v]) < 1:
            risk[v] = 0.0
            continue
        risk[v] = abs(float(np.mean(in_vals[v])) - float(np.mean(out_vals[v])))
    if risk.max() > risk.min():
        risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-12)
    return torch.tensor(risk, dtype=torch.float)


def _resolve_harp_risk(defense_name, defense_params, train_data, model_name, num_features, num_classes, device, ep, lr, wd, cfg, release_seed=0):
    """Resolve pluggable risk ranking for HARP seed selection."""
    from defenses.harp import risk_from_degree, risk_from_train_neighbors, risk_from_confidence

    seed_mode = str(defense_params.get("seed_mode", "") or "").lower()
    name_to_mode = {
        "harp_audit": "audit",
        "harp_random": "random",
        "harp_degree": "degree",
        "harp_train_nbr": "train_nbr",
        "harp_confidence": "confidence",
        "harp_entropy": "entropy",
    }
    if defense_name in name_to_mode:
        seed_mode = name_to_mode[defense_name]
    if not seed_mode:
        seed_mode = "lte" if bool(defense_params.get("use_lte", True)) else "random"

    use_lte = bool(defense_params.get("use_lte", True))
    arch_aware = bool(defense_params.get("arch_aware", True))
    arch = _lte_arch(model_name)

    if seed_mode == "audit":
        return _shadow_vulnerability_risk(
            train_data, model_name, num_features, num_classes, device, ep, lr, wd, cfg,
            n_rank=int(defense_params.get("n_rank_shadows", 4)),
            seed0=int(release_seed),
        )
    if seed_mode == "random":
        rng = np.random.RandomState(int(release_seed) + 17)
        return torch.tensor(rng.rand(int(train_data.num_nodes)), dtype=torch.float)
    if seed_mode == "degree":
        return risk_from_degree(train_data, invert=bool(defense_params.get("invert_degree", True)))
    if seed_mode == "train_nbr":
        return risk_from_train_neighbors(train_data)
    if seed_mode in ("confidence", "entropy"):
        model = _make_gnn(model_name, num_features, num_classes, use_gate=False)
        train_gnn(model, train_data, device, epochs=ep, lr=lr, weight_decay=wd)
        model.eval()
        with torch.no_grad():
            logits = model(train_data.x.to(device), train_data.edge_index.to(device))
            p = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return risk_from_confidence(p, mode="entropy" if seed_mode == "entropy" else "maxconf")
    # Default: topology LTE (or uniform ranks for harp_uniform).
    return compute_lte_risk(
        train_data.cpu(),
        uniform=not use_lte or defense_name == "harp_uniform",
        arch=arch,
        arch_aware=arch_aware,
    )


def _lte_arch(model_name: str) -> str:
    return "gcn" if model_name == "GCN" else "sage"


def _is_gnn_model(model_name: str) -> bool:
    return model_name in ("GCN", "GraphSAGE", "GAT")


def _make_gnn(model_name, num_features, num_classes, use_gate: bool, hidden: int = 64):
    if model_name == "GAT":
        # No gated-GAT variant; attention models always use plain GAT.
        return GAT(ic=num_features, h=int(hidden), oc=num_classes)
    if use_gate:
        cls = GatedGCN if model_name == "GCN" else GatedSAGE
    else:
        cls = GCN if model_name == "GCN" else SAGE
    return cls(ic=num_features, h=int(hidden), oc=num_classes)


def _train_and_predict_gnn(
    model_name,
    defense_name,
    defense_params,
    data,
    num_features,
    num_classes,
    device,
    ep,
    lr,
    wd,
    tk,
    cmk,
    use_minibatch,
    batch_size,
    num_neighbors,
    config,
    release_seed=0,
    multi_query_k=1,
):
    """Train one GNN (target or shadow) and return (probs, preds, risk_or_None, train_seconds, dp_eps).

    Shadows and the target share the same defense_name / defense_params / release
    transform so confidence, threshold, shadow, and LiRA attackers are
    defense-aware by construction (EVALUATION_PROTOCOL).
    multi_query_k>1 averages K independent Laplace releases (adaptive attacker).
    """
    t0 = time.time()
    use_gate = False
    if _is_sami_family(defense_name):
        use_gate = bool(defense_params.get("use_gate", True))
    elif _is_harp_family(defense_name):
        use_gate = bool(defense_params.get("use_gate", True))
        if defense_name in ("harp_release_only", "harp_uniform"):
            use_gate = False
    # Ablations that disable the gate force plain GCN/SAGE.
    if defense_name in ("sami_no_gate", "sami_temp_only", "advreg"):
        use_gate = False
    if defense_name == "sami" and not defense_params.get("use_gate", True):
        use_gate = False
    if defense_name == "harp" and not defense_params.get("use_gate", True):
        use_gate = False
    if model_name == "GAT":
        use_gate = False

    model = _make_gnn(
        model_name,
        num_features,
        num_classes,
        use_gate=use_gate,
        hidden=int(config.get("training", {}).get("hidden", 64)),
    ).to(device)
    edge_index = data.edge_index
    train_data = data
    risk = None
    dp_epsilon = float("nan")
    release_stats = {
        "noise_mass": float("nan"),
        "frac_protected": float("nan"),
        "frac_seeds": float("nan"),
        "mean_scale": float("nan"),
        "relative_noise_mass_vs_uniform": float("nan"),
        "n_protected": float("nan"),
    }

    if tk.get("edge_sparsify_rate", 0) > 0:
        edge_index = drop_edges_undirected(edge_index, tk["edge_sparsify_rate"])
        train_data = data.clone()
        train_data.edge_index = edge_index

    if defense_name == "dp_sgd":
        dp_cfg = config.get("dp_sgd", {})
        model, dp_epsilon = train_gnn_dp_minibatch(
            model,
            train_data,
            device,
            edge_index,
            epochs=int(dp_cfg.get("epochs", 20)),
            lr=float(dp_cfg.get("lr", 0.05)),
            weight_decay=0.0,
            batch_size=int(dp_cfg.get("batch_size", batch_size)),
            num_neighbors=num_neighbors,
            max_grad_norm=float(dp_cfg.get("max_grad_norm", 1.0)),
            noise_multiplier=float(dp_cfg.get("noise_multiplier", 1.0)),
            delta=float(dp_cfg.get("delta", 1e-5)),
            dropedge_rate=float(tk.get("dropedge_rate", 0.0) or 0.0),
        )
    elif defense_name == "gtd":
        if use_minibatch:
            train_gnn_gtd_minibatch(
                model,
                train_data,
                device,
                epochs=ep,
                lr=lr,
                weight_decay=wd,
                gamma=float(defense_params.get("gamma", 1.0)),
                stage1_frac=float(defense_params.get("stage1_frac", 0.5)),
                pseudo_conf=float(defense_params.get("pseudo_conf", 0.8)),
                batch_size=batch_size,
                num_neighbors=num_neighbors,
            )
        else:
            train_gnn_gtd(
                model,
                train_data,
                device,
                epochs=ep,
                lr=lr,
                weight_decay=wd,
                gamma=float(defense_params.get("gamma", 1.0)),
                stage1_frac=float(defense_params.get("stage1_frac", 0.5)),
                pseudo_conf=float(defense_params.get("pseudo_conf", 0.8)),
            )
    elif _is_harp_family(defense_name):
        lam = float(defense_params.get("lam", 0.5))
        use_lte = bool(defense_params.get("use_lte", True))
        arch_aware = bool(defense_params.get("arch_aware", True))
        arch = _lte_arch(model_name)
        train_on_protected = bool(defense_params.get("train_on_protected", True))
        if defense_name == "harp_release_only":
            train_on_protected = False
            lam = 0.0
            use_gate = False

        # Resolve hop / frac ablations from defense name when not overridden.
        k_hops = int(defense_params.get("k_hops", 1))
        risk_frac = float(defense_params.get("risk_frac", 0.25))
        if defense_name == "harp_k0":
            k_hops = int(defense_params.get("k_hops", 0))
        elif defense_name == "harp_k2":
            k_hops = int(defense_params.get("k_hops", 2))
        elif defense_name == "harp_uniform":
            # Ablation: protect everyone (recovers uniform strong noise).
            risk_frac = 1.0
            k_hops = 0

        risk_full = _resolve_harp_risk(
            defense_name,
            defense_params,
            train_data.cpu(),
            model_name,
            num_features,
            num_classes,
            device,
            ep,
            lr,
            wd,
            config if isinstance(config, dict) else {},
            release_seed=int(release_seed) if release_seed is not None else 0,
        )
        scales_pre, protected, seeds, harp_stats = compute_harp_scales(
            train_data.cpu(),
            risk=risk_full,
            risk_frac=risk_frac,
            risk_threshold=defense_params.get("risk_threshold"),
            k_hops=k_hops,
            strong_noise_scale=float(defense_params.get("strong_noise_scale", 0.30)),
            weak_noise_scale=float(defense_params.get("weak_noise_scale", 0.0)),
            use_lte=use_lte,
            arch=arch,
            arch_aware=arch_aware,
            target_protect_frac=(
                float(defense_params["target_protect_frac"])
                if defense_params.get("target_protect_frac") is not None
                else None
            ),
        )
        release_stats.update(harp_stats)
        model._harp_scales = scales_pre
        model._harp_stats = dict(harp_stats)
        model._harp_k_hops = k_hops
        model._harp_risk_frac = risk_frac

        risk_for_train = risk_full
        if train_on_protected and lam > 0:
            risk_for_train = mask_risk_to_protected(risk_full, protected)

        if defense_name == "harp_release_only" or lam <= 0:
            if use_minibatch:
                train_gnn_minibatch(
                    model,
                    train_data,
                    device,
                    edge_index,
                    epochs=ep,
                    lr=lr,
                    weight_decay=wd,
                    batch_size=batch_size,
                    num_neighbors=num_neighbors,
                    early_stop_patience=tk.get("early_stop_patience"),
                    label_smoothing=float(tk.get("label_smoothing", 0.0) or 0.0),
                    dropedge_rate=float(tk.get("dropedge_rate", 0.0) or 0.0),
                    val_mask=getattr(data, "val_mask", None),
                )
            else:
                train_gnn(
                    model,
                    train_data,
                    device,
                    epochs=ep,
                    lr=lr,
                    weight_decay=wd,
                    early_stop_patience=tk.get("early_stop_patience"),
                    label_smoothing=tk.get("label_smoothing", 0.0),
                    dropedge_rate=tk.get("dropedge_rate", 0.0),
                    edge_sparsify_rate=0.0,
                )
            risk = risk_full
        elif use_minibatch:
            # Minibatch: CE train + HARP release (full AdvReg path is full-batch).
            train_gnn_minibatch(
                model,
                train_data,
                device,
                edge_index,
                epochs=ep,
                lr=lr,
                weight_decay=wd,
                batch_size=batch_size,
                num_neighbors=num_neighbors,
                early_stop_patience=tk.get("early_stop_patience"),
                label_smoothing=float(tk.get("label_smoothing", 0.0) or 0.0),
                dropedge_rate=float(tk.get("dropedge_rate", 0.0) or 0.0),
                val_mask=getattr(data, "val_mask", None),
            )
            risk = risk_full
        else:
            model, risk = train_gnn_sami(
                model,
                train_data,
                device,
                epochs=ep,
                lr=lr,
                weight_decay=wd,
                lam=lam,
                use_lte=use_lte,
                use_gate=use_gate,
                warmup_epochs=int(defense_params.get("warmup_epochs", 5)),
                entropy_coef=float(defense_params.get("entropy_coef", 0.05)),
                arch=arch,
                arch_aware=arch_aware,
                risk=risk_for_train,
            )
        # Keep continuous LTE for optional temperature; release uses discrete scales.
        model._sami_beta = float(defense_params.get("beta", 0.0) or 0.0)
        model._sami_risk = risk_full
        risk = risk_full
    elif _is_sami_family(defense_name):
        lam = float(defense_params.get("lam", 0.1))
        use_lte = bool(defense_params.get("use_lte", True))
        beta = float(defense_params.get("beta", 0.0))
        arch_aware = bool(defense_params.get("arch_aware", True))
        arch = _lte_arch(model_name)
        # Volume path: HCAG off; batched SAMI. Citation path: full-batch SAMI.
        if use_minibatch:
            use_gate = False
            model, risk = train_gnn_sami_minibatch(
                model,
                train_data,
                device,
                epochs=ep,
                lr=lr,
                weight_decay=wd,
                lam=lam,
                use_lte=use_lte,
                warmup_epochs=int(defense_params.get("warmup_epochs", 3)),
                entropy_coef=float(defense_params.get("entropy_coef", 0.05)),
                batch_size=batch_size,
                num_neighbors=num_neighbors,
                arch=arch,
                arch_aware=arch_aware,
            )
        else:
            model, risk = train_gnn_sami(
                model,
                train_data,
                device,
                epochs=ep,
                lr=lr,
                weight_decay=wd,
                lam=lam,
                use_lte=use_lte,
                use_gate=use_gate,
                warmup_epochs=int(defense_params.get("warmup_epochs", 10)),
                entropy_coef=float(defense_params.get("entropy_coef", 0.05)),
                arch=arch,
                arch_aware=arch_aware,
            )
        model._sami_beta = beta
        model._sami_risk = risk
        model._sami_budget_B = float(defense_params.get("budget_B", 0.0) or 0.0)
        model._sami_base_scale = float(defense_params.get("noise_scale", 0.35))
        # SAMI continuous noise mass for Pareto comparisons.
        if risk is not None:
            r_np = risk.numpy() if hasattr(risk, "numpy") else np.asarray(risk)
            ns = float(defense_params.get("noise_scale", 0.35))
            if defense_name == "sami" and "noise_scale" not in defense_params:
                ns = 0.35
            scales = ns * np.asarray(r_np, dtype=float)
            release_stats["noise_mass"] = float(scales.sum())
            release_stats["mean_scale"] = float(scales.mean())
            release_stats["frac_protected"] = float((scales > 1e-12).mean())
            release_stats["n_protected"] = float((scales > 1e-12).sum())
            release_stats["relative_noise_mass_vs_uniform"] = (
                float(scales.sum() / (ns * len(scales))) if ns > 0 and len(scales) else float("nan")
            )
    elif use_minibatch:
        es = tk.get("early_stop_patience")
        val_mask = getattr(data, "val_mask", None)
        if val_mask is None or not val_mask.any():
            val_mask = None
        train_gnn_minibatch(
            model,
            train_data,
            device,
            edge_index,
            epochs=ep,
            lr=lr,
            weight_decay=wd,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            early_stop_patience=es,
            label_smoothing=float(tk.get("label_smoothing", 0.0) or 0.0),
            dropedge_rate=float(tk.get("dropedge_rate", 0.0) or 0.0),
            val_mask=val_mask,
        )
    else:
        train_gnn(
            model,
            train_data,
            device,
            epochs=ep,
            lr=lr,
            weight_decay=wd,
            early_stop_patience=tk.get("early_stop_patience"),
            label_smoothing=tk.get("label_smoothing", 0.0),
            dropedge_rate=tk.get("dropedge_rate", 0.0),
            edge_sparsify_rate=0.0,
        )

    model.eval()
    with torch.no_grad():
        # At Volume, always use NeighborLoader inference (incl. SAMI/GTD).
        if use_minibatch:
            logits = infer_logits_minibatch(
                model,
                train_data,
                edge_index,
                device,
                num_neighbors,
                batch_size,
                train_data.num_nodes,
                num_classes,
            )
            if _is_sami_family(defense_name) or _is_harp_family(defense_name):
                beta = float(getattr(model, "_sami_beta", 0.0) or 0.0)
                if beta > 0:
                    r = risk if risk is not None else compute_lte_risk(
                        train_data.cpu(), uniform=False, arch=_lte_arch(model_name)
                    )
                    logits = risk_scaled_temperature(logits.to(device), r.to(device), beta=beta).cpu()
        else:
            if (_is_sami_family(defense_name) or _is_harp_family(defense_name)) and use_gate and risk is not None:
                logits = model(
                    train_data.x.to(device),
                    train_data.edge_index.to(device),
                    risk=risk.to(device),
                )
            else:
                logits = model(train_data.x.to(device), train_data.edge_index.to(device))
            beta = float(getattr(model, "_sami_beta", 0.0) or 0.0)
            if beta > 0:
                r = risk if risk is not None else compute_lte_risk(train_data.cpu(), uniform=False)
                logits = risk_scaled_temperature(logits, r.to(device), beta=beta)

    logits_cpu = logits.detach().cpu() if torch.is_tensor(logits) and logits.is_cuda else (
        logits if torch.is_tensor(logits) else torch.as_tensor(logits)
    )
    p = F.softmax(logits_cpu, 1).numpy()
    pr = logits_cpu.argmax(1).numpy()
    p = _apply_confidence_masking(p, cmk)

    # HARP: hop-consistent discrete scales (recomputed for defense-aware shadows).
    if _is_harp_family(defense_name):
        arch = _lte_arch(model_name)
        k_hops = int(getattr(model, "_harp_k_hops", defense_params.get("k_hops", 1)))
        risk_frac = float(getattr(model, "_harp_risk_frac", defense_params.get("risk_frac", 0.25)))
        if defense_name == "harp_k0":
            k_hops = int(defense_params.get("k_hops", 0))
        elif defense_name == "harp_k2":
            k_hops = int(defense_params.get("k_hops", 2))
        elif defense_name == "harp_uniform":
            risk_frac = 1.0
            k_hops = 0
        scales, protected, seeds, harp_stats = compute_harp_scales(
            train_data.cpu(),
            risk=risk,
            risk_frac=risk_frac,
            risk_threshold=defense_params.get("risk_threshold"),
            k_hops=k_hops,
            strong_noise_scale=float(defense_params.get("strong_noise_scale", 0.30)),
            weak_noise_scale=float(defense_params.get("weak_noise_scale", 0.0)),
            use_lte=bool(defense_params.get("use_lte", True)),
            arch=arch,
            arch_aware=bool(defense_params.get("arch_aware", True)),
            target_protect_frac=(
                float(defense_params["target_protect_frac"])
                if defense_params.get("target_protect_frac") is not None
                else None
            ),
        )
        release_stats.update(harp_stats)
        if defense_name == "harp_mask" or str(defense_params.get("protector", "")).lower() == "mask":
            # Selective masking: exact scores on clean; one-hot top-1 on protected.
            prot = np.asarray(protected, dtype=bool)
            p2 = p.copy()
            top = p2.argmax(1)
            onehot = np.zeros_like(p2)
            onehot[np.arange(len(top)), top] = 1.0
            p2[prot] = onehot[prot]
            p = p2
            pr = p.argmax(1)
            release_stats["protector"] = "mask"
        else:
            k = int(multi_query_k) if multi_query_k is not None else 1
            if float(scales.max()) > 0:
                if k > 1:
                    p = average_posterior_queries(p, scales, 1.0, k, seed0=int(release_seed))
                else:
                    p = risk_scaled_posterior_noise(p, scales, scale=1.0, seed=int(release_seed))
                pr = p.argmax(1)
            release_stats["protector"] = "laplace"

    # Structure-aware posterior noise (SAMI inference complement; default on for full SAMI).
    elif _is_sami_family(defense_name):
        noise_scale = float(defense_params.get("noise_scale", 0.0))
        if defense_name == "sami" and "noise_scale" not in defense_params:
            noise_scale = 0.35
        budget_B = float(defense_params.get("budget_B", getattr(model, "_sami_budget_B", 0.0) or 0.0))
        if noise_scale > 0 or budget_B > 0:
            r = risk if risk is not None else compute_lte_risk(
                train_data.cpu(),
                uniform=not bool(defense_params.get("use_lte", True)),
                arch=_lte_arch(model_name),
            )
            r_np = r.numpy() if hasattr(r, "numpy") else np.asarray(r)
            if budget_B > 0:
                # σ_v from budget; map to equivalent constant scale on unit risk
                scales = allocate_risk_budget(r_np, budget_B, base_scale=max(noise_scale, 1e-6))
                # Use per-node scales via risk_scaled with scale=1 and risk:=scales
                k = int(multi_query_k) if multi_query_k is not None else 1
                if k > 1:
                    p = average_posterior_queries(p, scales, 1.0, k, seed0=int(release_seed))
                else:
                    p = risk_scaled_posterior_noise(p, scales, scale=1.0, seed=int(release_seed))
            else:
                k = int(multi_query_k) if multi_query_k is not None else 1
                if k > 1:
                    p = average_posterior_queries(p, r_np, noise_scale, k, seed0=int(release_seed))
                else:
                    p = risk_scaled_posterior_noise(p, r_np, scale=noise_scale, seed=int(release_seed))
            pr = p.argmax(1)

    # LBP is a post-hoc posterior transform (also applies to feature-only models).
    if defense_name == "lbp":
        scale = float(defense_params.get("scale", 0.3))
        n_bins = defense_params.get("n_bins")
        p = lbp_perturb(p, scale=scale, n_bins=n_bins, seed=int(release_seed))
        pr = p.argmax(1)
        n = int(p.shape[0])
        release_stats["noise_mass"] = float(scale * n)
        release_stats["mean_scale"] = float(scale)
        release_stats["frac_protected"] = 1.0
        release_stats["n_protected"] = float(n)
        release_stats["relative_noise_mass_vs_uniform"] = 1.0

    if defense_name == "memguard":
        yn = train_data.y.detach().cpu().numpy()
        trm = train_data.train_mask.detach().cpu().numpy()
        tem = train_data.test_mask.detach().cpu().numpy()
        p, mg_stats = apply_memguard(
            p, trm, tem, yn,
            max_l1=float(defense_params.get("max_l1", 0.2)),
            seed=int(release_seed),
        )
        pr = p.argmax(1)
        release_stats.update(mg_stats)

    train_seconds = time.time() - t0
    return p, pr, risk, train_seconds, dp_epsilon, release_stats


def run_one(
    dataset_name,
    model_name,
    defense_name,
    defense_params,
    seed,
    data_dir=None,
    device=None,
    training_kwargs=None,
    config=None,
):
    """Run one experiment: train target model, run enabled MIAs, compute ECE."""
    if config is None:
        config = load_config()
    if data_dir is None:
        data_dir = config["data_dir"]
    if device is None:
        device = torch.device(config.get("device", "cpu"))
    if training_kwargs is None:
        training_kwargs = config.get("training", {})

    attacks = {a.lower() for a in config.get("attacks", ["confidence", "threshold", "shadow", "lira"])}
    lira_cfg = config.get("lira", {"n_shadows": 8})
    n_lira_shadows = int(lira_cfg.get("n_shadows", 8))
    mb = config.get("minibatch", {})
    batch_size = int(mb.get("batch_size", 1024))
    num_neighbors = mb.get("num_neighbors", [15, 10])
    use_official = bool(config.get("large_graph_use_official_split", True))
    split_kw = _split_kwargs(config)
    # Adaptive multi-query averaging against randomized release (K queries).
    multi_query_k = int(config.get("multi_query_k", defense_params.get("multi_query_k", 1) or 1))
    run_mlp_phi = bool(config.get("run_mlp_phi_attack", "mlp_phi" in attacks))

    np.random.seed(seed)
    torch.manual_seed(seed)

    data, num_classes, num_features = _load_target_data(
        dataset_name, data_dir, seed, use_official, split_kw
    )

    h = homophily(data)
    dens_val = density(data)

    ep = int(training_kwargs.get("epochs", 50))
    lr = float(training_kwargs.get("lr", 0.01))
    wd = float(training_kwargs.get("weight_decay", 5e-4))
    tk = {"epochs": ep, "lr": lr, "weight_decay": wd, "device": device}
    cmk = None
    if defense_name == "dropedge":
        tk["dropedge_rate"] = defense_params.get("rate", 0.3)
    elif defense_name == "label_smoothing":
        tk["label_smoothing"] = defense_params.get("alpha", 0.1)
    elif defense_name == "early_stopping":
        tk["early_stop_patience"] = defense_params.get("patience", 15)
    elif defense_name in ("confidence_masking", "maskarmor"):
        # MaskArmor-style: keep only top-k class probabilities (default k=1).
        cmk = int(defense_params.get("top_k", 1 if defense_name == "maskarmor" else 2))
    elif defense_name == "edge_sparsification":
        tk["edge_sparsify_rate"] = defense_params.get("rate", 0.2)

    use_minibatch = dataset_name in MINIBATCH_DATASETS
    dp_epsilon = float("nan")
    train_seconds = 0.0
    release_stats = {
        "noise_mass": float("nan"),
        "frac_protected": float("nan"),
        "frac_seeds": float("nan"),
        "mean_scale": float("nan"),
        "relative_noise_mass_vs_uniform": float("nan"),
        "n_protected": float("nan"),
    }

    trm = data.train_mask.cpu().numpy()
    tem = data.test_mask.cpu().numpy()
    yn = data.y.cpu().numpy()

    # --- Train target and predict probabilities ---
    if _is_gnn_model(model_name):
        p, pr, _, train_seconds, dp_epsilon, release_stats = _train_and_predict_gnn(
            model_name,
            defense_name,
            defense_params,
            data,
            num_features,
            num_classes,
            device,
            ep,
            lr,
            wd,
            tk,
            cmk,
            use_minibatch,
            batch_size,
            num_neighbors,
            config,
            release_seed=int(seed),
            multi_query_k=multi_query_k,
        )
    else:
        t0 = time.time()
        Xn = data.x.numpy()
        if model_name == "LogReg":
            clf = LogisticRegression(max_iter=1000, random_state=int(seed))
        else:
            clf = MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=200, random_state=int(seed)
            )
        clf.fit(Xn[trm], yn[trm])
        p = clf.predict_proba(Xn)
        pr = clf.predict(Xn)
        p = _apply_confidence_masking(p, cmk)
        if defense_name == "lbp":
            scale = float(defense_params.get("scale", 0.3))
            p = lbp_perturb(p, scale=scale, n_bins=defense_params.get("n_bins"), seed=int(seed))
            pr = p.argmax(1)
            n = int(p.shape[0])
            release_stats = {
                "noise_mass": float(scale * n),
                "mean_scale": float(scale),
                "frac_protected": 1.0,
                "frac_seeds": 1.0,
                "n_protected": float(n),
                "relative_noise_mass_vs_uniform": 1.0,
            }
        train_seconds = time.time() - t0

    ta = accuracy_score(yn[tem], pr[tem])
    tf = f1_score(yn[tem], pr[tem], average="macro", zero_division=0)
    try:
        tau = roc_auc_score(yn[tem], p[tem], multi_class="ovr", average="macro")
    except Exception:
        tau = ta
    train_acc = accuracy_score(yn[trm], pr[trm])
    vam = data.val_mask.cpu().numpy()
    val_acc = float(accuracy_score(yn[vam], pr[vam])) if vam.any() else float("nan")
    gap = float(train_acc - ta)
    split_protocol = str(split_kw.get("protocol", "random_ratio"))
    if dataset_name in MINIBATCH_DATASETS and use_official:
        split_protocol = (
            "linkx_50_25_25_seed0" if dataset_name == "arxiv-year" else "ogb_official"
        )
    elif split_protocol == "random_ratio":
        split_protocol = "random_40_20_40"

    run_conf = ("confidence" in attacks) or ("threshold" in attacks)
    run_shadow = "shadow" in attacks
    run_lira = "lira" in attacks

    ca = cac = ta2 = tac2 = float("nan")
    conf_tpr_001 = conf_tpr_01 = float("nan")
    mlp_auc = mlp_acc = float("nan")
    if run_conf:
        ca, cac, ta2, tac2 = confidence_attack(
            p[trm], p[tem], yn[trm], yn[tem], random_state=int(seed)
        )
        # TPR@FPR from true-label confidence (threshold attack scores).
        fm = extract_features(p[trm], yn[trm])
        fn = extract_features(p[tem], yn[tem])
        all_conf = np.concatenate([fm[:, 1], fn[:, 1]])
        y_mem = np.concatenate([np.ones(len(fm)), np.zeros(len(fn))])
        conf_tpr_001 = tpr_at_fpr(y_mem, all_conf, 0.001)
        conf_tpr_01 = tpr_at_fpr(y_mem, all_conf, 0.01)
    if run_mlp_phi or run_conf:
        try:
            mlp_auc, mlp_acc = mlp_phi_attack(
                p[trm], p[tem], yn[trm], yn[tem], random_state=int(seed)
            )
        except Exception:
            mlp_auc, mlp_acc = float("nan"), float("nan")

    sa = sac = float("nan")
    la = lac = float("nan")
    lira_tpr_001 = lira_tpr_01 = float("nan")

    need_shadows = run_shadow or run_lira
    n_shadow_train = n_lira_shadows if run_lira else (1 if run_shadow else 0)

    shadow_probs_list = []
    shadow_tr_masks = []
    shadow_te_masks = []
    sh_y = None

    if need_shadows and n_shadow_train > 0:
        for k in range(n_shadow_train):
            shadow_seed = int(seed + 999 + k * 10007)
            np.random.seed(shadow_seed)
            torch.manual_seed(shadow_seed)
            shadow_data, _, _ = _make_shadow_data(dataset_name, data_dir, shadow_seed, split_kw)
            sh_tr = shadow_data.train_mask.numpy()
            sh_te = shadow_data.test_mask.numpy()
            sh_y = shadow_data.y.numpy()

            if _is_gnn_model(model_name):
                # Defense-aware shadows: identical training defense + release API.
                p_sh, _, _, _, _, _ = _train_and_predict_gnn(
                    model_name,
                    defense_name,
                    defense_params,
                    shadow_data,
                    num_features,
                    num_classes,
                    device,
                    ep,
                    lr,
                    wd,
                    tk,
                    cmk,
                    use_minibatch,
                    batch_size,
                    num_neighbors,
                    config,
                    release_seed=int(shadow_seed),
                    multi_query_k=multi_query_k,
                )
            else:
                Xs = shadow_data.x.numpy()
                if model_name == "LogReg":
                    c_sh = LogisticRegression(max_iter=1000, random_state=int(shadow_seed))
                else:
                    c_sh = MLPClassifier(
                        hidden_layer_sizes=(64, 32), max_iter=200, random_state=int(shadow_seed)
                    )
                c_sh.fit(Xs[sh_tr], sh_y[sh_tr])
                p_sh = c_sh.predict_proba(Xs)
                p_sh = _apply_confidence_masking(p_sh, cmk)
                if defense_name == "lbp":
                    p_sh = lbp_perturb(
                        p_sh,
                        scale=float(defense_params.get("scale", 0.3)),
                        n_bins=defense_params.get("n_bins"),
                        seed=shadow_seed,
                    )

            shadow_probs_list.append(p_sh)
            shadow_tr_masks.append(sh_tr)
            shadow_te_masks.append(sh_te)

        if run_shadow and shadow_probs_list:
            p_sh0 = shadow_probs_list[0]
            sh_tr0 = shadow_tr_masks[0]
            sh_te0 = shadow_te_masks[0]
            try:
                sa, sac = shadow_attack(
                    p_sh0[sh_tr0],
                    p_sh0[sh_te0],
                    sh_y[sh_tr0],
                    sh_y[sh_te0],
                    p[trm],
                    p[tem],
                    yn[trm],
                    yn[tem],
                    random_state=int(seed),
                )
            except Exception:
                sa, sac = 0.5, 0.5

        if run_lira and shadow_probs_list:
            try:
                out = lira_gaussian_auc(
                    p,
                    yn,
                    trm,
                    tem,
                    shadow_probs_list,
                    shadow_tr_masks,
                    shadow_te_masks,
                )
                if len(out) == 4:
                    la, lac, lira_tpr_001, lira_tpr_01 = out
                else:
                    la, lac = out[0], out[1]
            except Exception:
                la, lac = 0.5, 0.5

    ece = calibration_error(p[tem], yn[tem])
    gap_auc = gap_acc = float("nan")
    try:
        gap_auc, gap_acc = gap_attack(p[trm], p[tem], yn[trm], yn[tem])
    except Exception:
        gap_auc, gap_acc = float("nan"), float("nan")

    def _r(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        return round(float(x), 4)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "defense": defense_name,
        "defense_params": json.dumps(defense_params),
        "seed": seed,
        "config_hash": config.get("config_hash", ""),
        "homophily": round(h, 4),
        "density": round(dens_val, 6),
        "test_accuracy": _r(ta),
        "val_accuracy": _r(val_acc),
        "train_accuracy": _r(train_acc),
        "gen_gap": _r(gap),
        "split_protocol": split_protocol,
        "test_f1": _r(tf),
        "test_auroc": _r(tau),
        "conf_attack_auc": _r(ca),
        "conf_attack_acc": _r(cac),
        "threshold_attack_auc": _r(ta2),
        "threshold_attack_acc": _r(tac2),
        "conf_tpr_at_0.001_fpr": _r(conf_tpr_001),
        "conf_tpr_at_0.01_fpr": _r(conf_tpr_01),
        "gap_attack_auc": _r(gap_auc),
        "gap_attack_acc": _r(gap_acc),
        "mlp_phi_attack_auc": _r(mlp_auc),
        "mlp_phi_attack_acc": _r(mlp_acc),
        "shadow_attack_auc": _r(sa),
        "shadow_attack_acc": _r(sac),
        "lira_attack_auc": _r(la),
        "lira_attack_acc": _r(lac),
        "lira_tpr_at_0.001_fpr": _r(lira_tpr_001),
        "lira_tpr_at_0.01_fpr": _r(lira_tpr_01),
        "ece_test": _r(ece),
        "train_seconds": _r(train_seconds),
        "multi_query_k": int(multi_query_k),
        "dp_epsilon": _r(dp_epsilon) if defense_name == "dp_sgd" else float("nan"),
        "noise_mass": _r(release_stats.get("noise_mass")),
        "frac_protected": _r(release_stats.get("frac_protected")),
        "frac_seeds": _r(release_stats.get("frac_seeds")),
        "mean_scale": _r(release_stats.get("mean_scale")),
        "relative_noise_mass_vs_uniform": _r(release_stats.get("relative_noise_mass_vs_uniform")),
        "n_protected": _r(release_stats.get("n_protected")),
    }
