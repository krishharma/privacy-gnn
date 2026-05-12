"""
Single-experiment runner: load data, train model, run MIAs and calibration, return metrics.

LiRA uses multiple shadow models; confidence/threshold/shadow attacks keep the 4D φ feature map.
Large OGB/Reddit graphs use NeighborLoader (see graph_minibatch.py).
"""
from __future__ import annotations

import json
import math
import os
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
)
from models import GCN, SAGE
from training import train_gnn
from attacks import confidence_attack, shadow_attack, calibration_error
from ogb_loader import MINIBATCH_DATASETS, load_large_benchmark
from graph_minibatch import train_gnn_minibatch, infer_logits_minibatch, train_gnn_dp_minibatch
from lira_attack import lira_gaussian_auc


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


def _load_target_data(dataset_name: str, data_dir: str, seed: int, use_official_large: bool):
    if dataset_name in ("Cora", "Citeseer"):
        data, num_classes, num_features = load_citation(dataset_name, data_dir=data_dir)
        return resplit(data, seed), num_classes, num_features
    if dataset_name.startswith("synthetic_"):
        parts = dataset_name.split("_")
        return make_synthetic(homo=parts[1], dens=parts[2], seed=seed)
    if dataset_name in MINIBATCH_DATASETS:
        if dataset_name == "Reddit":
            root = os.path.join(data_dir, "pyg")
        else:
            root = os.path.join(data_dir, "ogb")
        data, num_classes, num_features = load_large_benchmark(dataset_name, root)
        if use_official_large:
            return data, num_classes, num_features
        return resplit(data.clone(), seed), num_classes, num_features
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _make_shadow_data(dataset_name: str, data_dir: str, shadow_seed: int):
    if dataset_name in ("Cora", "Citeseer"):
        sd, nc, nf = load_citation(dataset_name, data_dir=data_dir)
        return resplit(sd, shadow_seed), nc, nf
    if dataset_name.startswith("synthetic_"):
        parts = dataset_name.split("_")
        return make_synthetic(homo=parts[1], dens=parts[2], seed=shadow_seed)
    if dataset_name in MINIBATCH_DATASETS:
        if dataset_name == "Reddit":
            root = os.path.join(data_dir, "pyg")
        else:
            root = os.path.join(data_dir, "ogb")
        data, nc, nf = load_large_benchmark(dataset_name, root)
        return resplit(data.clone(), shadow_seed), nc, nf
    raise ValueError(f"Unknown dataset: {dataset_name}")


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
    """
    Run one experiment: train target model, run enabled MIAs, compute ECE.
    """
    if config is None:
        config = load_config()
    if data_dir is None:
        data_dir = config["data_dir"]
    if device is None:
        device = torch.device(config.get("device", "cpu"))
    if training_kwargs is None:
        training_kwargs = config.get("training", {})

    attacks = {a.lower() for a in config.get("attacks", ["confidence", "threshold", "shadow", "lira"])}
    lira_cfg = config.get("lira", {"n_shadows": 3})
    n_lira_shadows = int(lira_cfg.get("n_shadows", 3))
    mb = config.get("minibatch", {})
    batch_size = int(mb.get("batch_size", 1024))
    num_neighbors = mb.get("num_neighbors", [15, 10])
    use_official = bool(config.get("large_graph_use_official_split", True))

    np.random.seed(seed)
    torch.manual_seed(seed)

    data, num_classes, num_features = _load_target_data(
        dataset_name, data_dir, seed, use_official
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
    elif defense_name == "confidence_masking":
        cmk = int(defense_params.get("top_k", 2))
    elif defense_name == "edge_sparsification":
        tk["edge_sparsify_rate"] = defense_params.get("rate", 0.2)

    use_minibatch = dataset_name in MINIBATCH_DATASETS
    dp_epsilon = float("nan")

    trm = data.train_mask.cpu().numpy()
    tem = data.test_mask.cpu().numpy()
    yn = data.y.cpu().numpy()

    # --- Train target and predict probabilities ---
    if model_name in ("GCN", "GraphSAGE"):
        model = (GCN if model_name == "GCN" else SAGE)(
            ic=num_features, h=64, oc=num_classes
        ).to(device)
        edge_index = data.edge_index
        train_data = data
        if tk.get("edge_sparsify_rate", 0) > 0:
            edge_index = drop_edges_undirected(edge_index, tk["edge_sparsify_rate"])
            train_data = data.clone()
            train_data.edge_index = edge_index

        if defense_name == "dp_sgd":
            dp_cfg = config.get("dp_sgd", {})
            dp_ep = int(dp_cfg.get("epochs", 20))
            dp_lr = float(dp_cfg.get("lr", 0.05))
            dp_bs = int(dp_cfg.get("batch_size", batch_size))
            dp_c = float(dp_cfg.get("max_grad_norm", 1.0))
            dp_nm = float(dp_cfg.get("noise_multiplier", 1.0))
            dp_delta = float(dp_cfg.get("delta", 1e-5))
            model, dp_epsilon = train_gnn_dp_minibatch(
                model,
                train_data,
                device,
                edge_index,
                epochs=dp_ep,
                lr=dp_lr,
                weight_decay=0.0,
                batch_size=dp_bs,
                num_neighbors=num_neighbors,
                max_grad_norm=dp_c,
                noise_multiplier=dp_nm,
                delta=dp_delta,
                dropedge_rate=float(tk.get("dropedge_rate", 0.0) or 0.0),
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
            else:
                logits = model(train_data.x.to(device), train_data.edge_index.to(device))
        p = F.softmax(logits, 1).cpu().numpy()
        pr = logits.argmax(1).cpu().numpy()
        p = _apply_confidence_masking(p, cmk)
    else:
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

    ta = accuracy_score(yn[tem], pr[tem])
    tf = f1_score(yn[tem], pr[tem], average="macro", zero_division=0)
    try:
        tau = roc_auc_score(yn[tem], p[tem], multi_class="ovr", average="macro")
    except Exception:
        tau = ta

    run_conf = ("confidence" in attacks) or ("threshold" in attacks)
    run_shadow = "shadow" in attacks
    run_lira = "lira" in attacks

    ca = cac = ta2 = tac2 = float("nan")
    if run_conf:
        ca, cac, ta2, tac2 = confidence_attack(
            p[trm], p[tem], yn[trm], yn[tem], random_state=int(seed)
        )

    sa = sac = float("nan")
    la = lac = float("nan")

    need_shadows = run_shadow or run_lira
    n_shadow_train = n_lira_shadows if run_lira else (1 if run_shadow else 0)

    shadow_probs_list = []
    shadow_tr_masks = []
    shadow_te_masks = []

    if need_shadows and n_shadow_train > 0:
        for k in range(n_shadow_train):
            shadow_seed = int(seed + 999 + k * 10007)
            np.random.seed(shadow_seed)
            torch.manual_seed(shadow_seed)
            shadow_data, _, _ = _make_shadow_data(dataset_name, data_dir, shadow_seed)
            sh_tr = shadow_data.train_mask.numpy()
            sh_te = shadow_data.test_mask.numpy()
            sh_y = shadow_data.y.numpy()

            if model_name in ("GCN", "GraphSAGE"):
                shadow_model = (GCN if model_name == "GCN" else SAGE)(
                    ic=num_features, h=64, oc=num_classes
                ).to(device)
                sh_ei = shadow_data.edge_index
                shadow_train_data = shadow_data
                if tk.get("edge_sparsify_rate", 0) > 0:
                    sh_ei = drop_edges_undirected(sh_ei, tk["edge_sparsify_rate"])
                    shadow_train_data = shadow_data.clone()
                    shadow_train_data.edge_index = sh_ei
                if defense_name == "dp_sgd":
                    dp_cfg = config.get("dp_sgd", {})
                    shadow_model, _ = train_gnn_dp_minibatch(
                        shadow_model,
                        shadow_train_data,
                        device,
                        sh_ei,
                        epochs=int(dp_cfg.get("epochs", 20)),
                        lr=float(dp_cfg.get("lr", 0.05)),
                        weight_decay=0.0,
                        batch_size=int(dp_cfg.get("batch_size", batch_size)),
                        num_neighbors=num_neighbors,
                        max_grad_norm=float(dp_cfg.get("max_grad_norm", 1.0)),
                        noise_multiplier=float(dp_cfg.get("noise_multiplier", 1.0)),
                        delta=float(dp_cfg.get("delta", 1e-5)),
                    )
                elif use_minibatch:
                    val_mask = getattr(shadow_data, "val_mask", None)
                    if val_mask is None or not val_mask.any():
                        val_mask = None
                    train_gnn_minibatch(
                        shadow_model,
                        shadow_train_data,
                        device,
                        sh_ei,
                        epochs=ep,
                        lr=lr,
                        weight_decay=wd,
                        batch_size=batch_size,
                        num_neighbors=num_neighbors,
                        early_stop_patience=tk.get("early_stop_patience"),
                        label_smoothing=float(tk.get("label_smoothing", 0.0) or 0.0),
                        dropedge_rate=float(tk.get("dropedge_rate", 0.0) or 0.0),
                        val_mask=val_mask,
                    )
                else:
                    train_gnn(
                        shadow_model,
                        shadow_train_data,
                        device,
                        epochs=ep,
                        lr=lr,
                        weight_decay=wd,
                        early_stop_patience=tk.get("early_stop_patience"),
                        label_smoothing=tk.get("label_smoothing", 0.0),
                        dropedge_rate=tk.get("dropedge_rate", 0.0),
                        edge_sparsify_rate=0.0,
                    )
                shadow_model.eval()
                with torch.no_grad():
                    if use_minibatch:
                        slogits = infer_logits_minibatch(
                            shadow_model,
                            shadow_train_data,
                            sh_ei,
                            device,
                            num_neighbors,
                            batch_size,
                            shadow_train_data.num_nodes,
                            num_classes,
                        )
                    else:
                        slogits = shadow_model(
                            shadow_train_data.x.to(device),
                            shadow_train_data.edge_index.to(device),
                        )
                p_sh = F.softmax(slogits, 1).cpu().numpy()
                p_sh = _apply_confidence_masking(p_sh, cmk)
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
                la, lac = lira_gaussian_auc(
                    p,
                    yn,
                    trm,
                    tem,
                    shadow_probs_list,
                    shadow_tr_masks,
                    shadow_te_masks,
                )
            except Exception:
                la, lac = 0.5, 0.5

    ece = calibration_error(p[tem], yn[tem])

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
        "homophily": round(h, 4),
        "density": round(dens_val, 6),
        "test_accuracy": _r(ta),
        "test_f1": _r(tf),
        "test_auroc": _r(tau),
        "conf_attack_auc": _r(ca),
        "conf_attack_acc": _r(cac),
        "threshold_attack_auc": _r(ta2),
        "threshold_attack_acc": _r(tac2),
        "shadow_attack_auc": _r(sa),
        "shadow_attack_acc": _r(sac),
        "lira_attack_auc": _r(la),
        "lira_attack_acc": _r(lac),
        "ece_test": _r(ece),
        "dp_epsilon": _r(dp_epsilon) if defense_name == "dp_sgd" else float("nan"),
    }
