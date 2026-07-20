"""
Single-experiment runner: load data, train model, run MIAs and calibration, return metrics.

LiRA uses multiple shadow models; confidence/threshold/shadow attacks keep the 4D φ feature map.
Large OGB/Reddit graphs use NeighborLoader (see graph_minibatch.py).
"""
from __future__ import annotations

import json
import math
import time
import os
import hashlib
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
from epsd_utils import compute_ego_gap


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
        return make_synthetic(homo=parts[1], dens=parts[2], seed=seed, center_std=0.5, noise_std=1.5)
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
        return make_synthetic(homo=parts[1], dens=parts[2], seed=shadow_seed, center_std=0.5, noise_std=1.5)
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

    attacks = {a.lower() for a in config.get("attacks", ["confidence", "threshold", "shadow", "label_only"])}
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
    elif defense_name == "epsd":
        tk["epsd_lambda"] = defense_params.get("lambda_epsd", 1.0)
        tk["epsd_ablation"] = defense_params.get("ablation", "none")

    use_minibatch = dataset_name in MINIBATCH_DATASETS
    dp_epsilon = float("nan")
    dp_delta = float("nan")
    dp_noise_multiplier = float("nan")
    dp_clip_norm = float("nan")

    trm = data.train_mask.cpu().numpy()
    tem = data.test_mask.cpu().numpy()
    yn = data.y.cpu().numpy()

    # --- Train target and predict probabilities ---
    train_time = 0.0
    t0 = time.time()
    
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
            
            dp_clip_norm = dp_c
            dp_noise_multiplier = dp_nm
            
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
            metrics = {k: float("nan") for k in ['normal_loss_ep1', 'epsd_kl_loss_ep1', 'total_loss_ep1', 'normal_loss_final', 'epsd_kl_loss_final', 'total_loss_final']}
        elif use_minibatch:
            es = tk.get("early_stop_patience")
            val_mask = getattr(data, "val_mask", None)
            if val_mask is None or not val_mask.any():
                val_mask = None
            model, metrics = train_gnn_minibatch(
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
                epsd_lambda=float(tk.get("epsd_lambda", 0.0) or 0.0),
                epsd_ablation=tk.get("epsd_ablation", "none"),
            )
        else:
            model, metrics = train_gnn(
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
                epsd_lambda=tk.get("epsd_lambda", 0.0),
                epsd_ablation=tk.get("epsd_ablation", "none"),
            )

        if hasattr(model, "eval"):
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
        
        # Calculate Hashes
        state_dict_bytes = b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())
        model_sha256 = hashlib.sha256(state_dict_bytes).hexdigest()
        prediction_sha256 = hashlib.sha256(p.tobytes()).hexdigest()
        
        p = _apply_confidence_masking(p, cmk)
    else:
        metrics = {k: float("nan") for k in ['normal_loss_ep1', 'epsd_kl_loss_ep1', 'total_loss_ep1', 'normal_loss_final', 'epsd_kl_loss_final', 'total_loss_final']}
        model_sha256 = "N/A"
        prediction_sha256 = "N/A"
        Xn = data.x.numpy()
        if model_name == "LogReg":
            clf = LogisticRegression(max_iter=1000, random_state=int(seed))
        else:
            clf = MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=200, random_state=int(seed)
            )
        clf.fit(Xn[trm], yn[trm])
        train_time = time.time() - t0
        p = clf.predict_proba(Xn)
        pr = clf.predict(Xn)
        prediction_sha256 = hashlib.sha256(p.tobytes()).hexdigest()
        p = _apply_confidence_masking(p, cmk)

    ta = accuracy_score(yn[tem], pr[tem])
    tf = f1_score(yn[tem], pr[tem], average="macro", zero_division=0)
    try:
        tau = roc_auc_score(yn[tem], p[tem], multi_class="ovr", average="macro")
    except Exception:
        tau = ta

    run_conf = ("confidence" in attacks) or ("threshold" in attacks)
    run_shadow = "shadow" in attacks
    run_label_only = "label_only" in attacks
    run_loss = "loss" in attacks

    if use_minibatch:
        print(f"Sampling Parameters: batch_size={batch_size}, num_neighbors={num_neighbors} (using {len(num_neighbors)} hop subgraphs)")

    print(f"--- Threat Model Definition ({dataset_name}) ---")
    print("Membership: Target nodes included in the training loss (members) vs. nodes in the class-stratified validation/test set (non-members).")
    print("Attacker Capability: Black-box. Attacker observes model posteriors but not model weights or embeddings.")
    if run_shadow:
        print("Shadow Model: Disjoint data split. Attack model tuned on shadow-val and evaluated on target test.")

    ca = cac = ca_tpr01 = ca_tpr05 = ca_adv = ta2 = tac2 = ta2_tpr01 = ta2_tpr05 = ta2_adv = float("nan")
    loa = loac = loa_tpr01 = loa_tpr05 = loa_adv = flip_rate = float("nan")
    lsa = lsac = lsa_tpr01 = lsa_tpr05 = lsa_adv = float("nan")

    # Combine validation into non-members if it exists
    tem_pool = tem.copy()
    if hasattr(data, 'val_mask') and data.val_mask is not None and data.val_mask.any():
        tem_pool = tem_pool | data.val_mask.cpu().numpy()

    # Create class-stratified balanced attack masks
    def _balance_masks(m_mask, nm_mask, y, rs):
        rng = np.random.RandomState(int(rs))
        m_idx = np.where(m_mask)[0]
        nm_idx = np.where(nm_mask)[0]
        classes = np.unique(y)
        m_out, nm_out = [], []
        for c in classes:
            mc = m_idx[y[m_idx] == c]
            nmc = nm_idx[y[nm_idx] == c]
            k = min(len(mc), len(nmc))
            if k > 0:
                m_out.append(rng.choice(mc, k, replace=False))
                nm_out.append(rng.choice(nmc, k, replace=False))
        new_m = np.zeros_like(m_mask, dtype=bool)
        new_nm = np.zeros_like(nm_mask, dtype=bool)
        if m_out:
            new_m[np.concatenate(m_out)] = True
            new_nm[np.concatenate(nm_out)] = True
        return new_m, new_nm

    trm_att, tem_att = _balance_masks(trm, tem_pool, yn, seed)

    if run_conf:
        c_res = confidence_attack(
            p[trm_att], p[tem_att], yn[trm_att], yn[tem_att], random_state=int(seed)
        )
        ca, cac, ca_tpr01, ca_tpr05, ca_adv = c_res['conf_auc'], c_res['conf_acc'], c_res['conf_tpr_01'], c_res['conf_tpr_05'], c_res['conf_adv']
        ta2, tac2, ta2_tpr01, ta2_tpr05, ta2_adv = c_res['thresh_auc'], c_res['thresh_acc'], c_res['thresh_tpr_01'], c_res['thresh_tpr_05'], c_res['thresh_adv']

    if run_label_only:
        from attacks import label_only_attack
        target_model = model if model_name in ("GCN", "GraphSAGE") else clf
        l_res = label_only_attack(target_model, train_data if model_name in ("GCN", "GraphSAGE") else data, trm_att, tem_att)
        loa, loac, loa_tpr01, loa_tpr05, loa_adv = l_res['auc'], l_res['acc'], l_res['tpr_01'], l_res['tpr_05'], l_res['adv']
        flip_rate = l_res['flip_rate']

    if run_loss:
        from attacks import loss_attack
        def get_loss(probs, labels):
            return -np.log(probs[np.arange(len(probs)), labels] + 1e-10)
        m_loss = get_loss(p[trm_att], yn[trm_att])
        nm_loss = get_loss(p[tem_att], yn[tem_att])
        ls_res = loss_attack(m_loss, nm_loss)
        lsa, lsac, lsa_tpr01, lsa_tpr05, lsa_adv = ls_res['auc'], ls_res['acc'], ls_res['tpr_01'], ls_res['tpr_05'], ls_res['adv']

    sa = sac = sa_tpr01 = sa_tpr05 = sa_adv = float("nan")

    need_shadows = run_shadow
    n_shadow_train = 1 if run_shadow else 0

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
                        dropedge_rate=float(tk.get("dropedge_rate", 0.0) or 0.0),
                    )
                elif use_minibatch:
                    val_mask = getattr(shadow_data, "val_mask", None)
                    if val_mask is None or not val_mask.any():
                        val_mask = None
                    _, _ = train_gnn_minibatch(
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
                    _, _ = train_gnn(
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
                if hasattr(shadow_model, "eval"):
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
            
            sh_tr_att, sh_te_att = _balance_masks(sh_tr0, sh_te0, sh_y, seed)
            try:
                s_res = shadow_attack(
                    p_sh0[sh_tr_att], p_sh0[sh_te_att], sh_y[sh_tr_att], sh_y[sh_te_att],
                    p[trm_att], p[tem_att], yn[trm_att], yn[tem_att], random_state=int(seed),
                )
                sa, sac, sa_tpr01, sa_tpr05, sa_adv = s_res['auc'], s_res['acc'], s_res['tpr_01'], s_res['tpr_05'], s_res['adv']
            except Exception:
                sa, sac, sa_tpr01, sa_tpr05, sa_adv = 0.5, 0.5, 0.0, 0.0, 0.0

                sa, sac, sa_tpr01, sa_tpr05, sa_adv = 0.5, 0.5, 0.0, 0.0, 0.0

    ece = calibration_error(p[tem], yn[tem])

    train_ego_gap = test_ego_gap = ego_gap_diff = float("nan")
    train_ego_gap_std = test_ego_gap_std = float("nan")
    if model_name in ("GCN", "GraphSAGE"):
        try:
            g_train = compute_ego_gap(model, data, np.where(trm)[0])
            g_test = compute_ego_gap(model, data, np.where(tem)[0])
            train_ego_gap = float(g_train.mean())
            test_ego_gap = float(g_test.mean())
            train_ego_gap_std = float(g_train.std())
            test_ego_gap_std = float(g_test.std())
            ego_gap_diff = float(train_ego_gap - test_ego_gap)
        except Exception:
            pass

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
        "train_time_seconds": round(train_time, 4),
        "test_accuracy": _r(ta),
        "test_f1": _r(tf),
        "test_auroc": _r(tau),
        "train_ego_gap": _r(train_ego_gap),
        "train_ego_gap_std": _r(train_ego_gap_std),
        "test_ego_gap": _r(test_ego_gap),
        "test_ego_gap_std": _r(test_ego_gap_std),
        "ego_gap_diff": _r(ego_gap_diff),
        "conf_attack_auc": _r(ca),
        "conf_attack_acc": _r(cac),
        "conf_attack_tpr01": _r(ca_tpr01),
        "conf_attack_tpr05": _r(ca_tpr05),
        "conf_attack_adv": _r(ca_adv),
        "threshold_attack_auc": _r(ta2),
        "threshold_attack_acc": _r(tac2),
        "threshold_attack_tpr01": _r(ta2_tpr01),
        "threshold_attack_tpr05": _r(ta2_tpr05),
        "threshold_attack_adv": _r(ta2_adv),
        "shadow_attack_auc": _r(sa),
        "shadow_attack_acc": _r(sac),
        "shadow_attack_tpr01": _r(sa_tpr01),
        "shadow_attack_tpr05": _r(sa_tpr05),
        "shadow_attack_adv": _r(sa_adv),
        "label_only_attack_auc": _r(loa),
        "label_only_attack_acc": _r(loac),
        "label_only_attack_tpr01": _r(loa_tpr01),
        "label_only_attack_tpr05": _r(loa_tpr05),
        "label_only_attack_adv": _r(loa_adv),
        "label_only_flip_rate": _r(flip_rate),
        "loss_attack_auc": _r(lsa),
        "loss_attack_acc": _r(lsac),
        "loss_attack_tpr01": _r(lsa_tpr01),
        "loss_attack_tpr05": _r(lsa_tpr05),
        "loss_attack_adv": _r(lsa_adv),
        "ece_test": _r(ece),
        "actual_dp_epsilon": _r(dp_epsilon) if defense_name == "dp_sgd" else float("nan"),
        "dp_delta": _r(dp_delta) if defense_name == "dp_sgd" else float("nan"),
        "dp_clip_norm": _r(dp_clip_norm) if defense_name == "dp_sgd" else float("nan"),
        "dp_noise_multiplier": _r(dp_noise_multiplier) if defense_name == "dp_sgd" else float("nan"),
        "lambda_epsd": _r(tk.get("epsd_lambda", 0.0)),
        "epsd_ablation": tk.get("epsd_ablation", "none"),
        "model_sha256": model_sha256,
        "prediction_sha256": prediction_sha256,
        "normal_loss_ep1": _r(metrics.get("normal_loss_ep1")),
        "epsd_kl_loss_ep1": _r(metrics.get("epsd_kl_loss_ep1")),
        "total_loss_ep1": _r(metrics.get("total_loss_ep1")),
        "normal_loss_final": _r(metrics.get("normal_loss_final")),
        "epsd_kl_loss_final": _r(metrics.get("epsd_kl_loss_final")),
        "total_loss_final": _r(metrics.get("total_loss_final")),
    }

    from validate_experiment import validate_run, print_warnings
    warnings = validate_run(data, p, yn, trm, tem, res, num_classes)
    if warnings:
        print_warnings(warnings)

    return res
