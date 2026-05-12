# PrivacyGNN: Privacy Leakage and Lightweight Defenses in Graph Neural Networks

**PI:** Dr. Shaikh Arifuzzaman  
**Student Researcher:** Krish Sharma  

This repository contains a reproducible, config-driven study of membership inference attacks (MIAs) against graph neural networks (GNNs). The project compares GCN and GraphSAGE against feature-only baselines, varies graph homophily and density in controlled synthetic graphs, evaluates Cora and Citeseer citation benchmarks, and tests lightweight defenses such as DropEdge, label smoothing, early stopping, confidence masking, and edge sparsification.

The main goal is to measure when supervised node-membership leakage appears, how it depends on graph structure and model architecture, and whether simple defenses reduce attack success without large utility loss.

---

## Project structure (easy to read and extend)

| File | Purpose |
|------|--------|
| **`config.py`** | Paths, device, and **config-driven** experiment grid (loads `experiment_config.yaml`). |
| **`experiment_config.yaml`** | Edit this to change datasets, models, defenses, seeds without touching code. |
| **`experiment_config_paper.yaml`** | Reproduces the paper figure grid: Cora, Citeseer, six synthetics; five seeds; confidence/threshold/shadow attacks only (no OGB/LiRA/dp_sgd). Run: `PRIVACYGNN_CONFIG=experiment_config_paper.yaml python run_final.py`. |
| **`models.py`** | GCN and GraphSAGE (two-layer) for node classification. |
| **`data.py`** | Load Cora/Citeseer; generate synthetic graphs (homophily/density); resplit; homophily/density stats. |
| **`ogb_loader.py`** | Stub in this repo (`MINIBATCH_DATASETS` empty). Extend with OGB loaders to enable `ogbn-arxiv` / large-graph configs. |
| **`graph_minibatch.py`** | Stub; neighbor-sampled training not active until implemented. |
| **`lira_attack.py`** | Placeholder `lira_gaussian_auc` (chance-level) when `attacks` includes `lira` without a full implementation. |
| **`training.py`** | GNN training with DropEdge, label smoothing, early stopping, edge sparsification. |
| **`attacks.py`** | Confidence-based MIA, threshold MIA, shadow-model MIA, and calibration error (ECE). |
| **`experiment.py`** | Single run: load data → train target (and shadows) → run enabled attacks → metrics row. |
| **`stats_utils.py`** | Exploratory paired t-tests; bootstrap CIs over seeds (separate from p-values). |
| **`run_final.py`** | Main entry: full grid → `all_results.csv`, `summary.csv`, `significance*.csv`, `summary_bootstrap.csv`. |
| **`generate_figures.py`** | Produce 4–6 publication figures from results. |
| **`fix_figures.py`** | Optional: improved labels for Fig 2 and Fig 4. |
| **`paper/build_manuscript.py`** | Build 4–6 page workshop manuscript PDF. |

---

## Setup

```bash
cd privacy-gnn
pip install -r requirements.txt
```

---

## Run experiments (config-driven)

```bash
python run_final.py
# Paper / figures (Cora + Citeseer + synthetics, no OGB):
# PRIVACYGNN_CONFIG=experiment_config_paper.yaml python run_final.py
```

- Reads **`experiment_config.yaml`** (or override with **`PRIVACYGNN_CONFIG`**) for datasets, models, defenses, seeds.
- Writes **`results/all_results.csv`**, **`results/summary.csv`**, **`results/significance.csv`** (exploratory t-tests on conf AUC), **`results/significance_lira.csv`** (t-tests on LiRA AUC if present), **`results/summary_bootstrap.csv`** (bootstrap CIs over seeds; not p-values).
- YAML keys: `attacks`, `lira.n_shadows`, `bootstrap`, `minibatch`, `large_graph_use_official_split`, `dp_sgd`, `dp_sgd_datasets`, `device`.

---

## Generate figures and manuscript

```bash
python generate_figures.py    # → figures/fig1_*.png, fig2_*, ...
python paper/build_manuscript.py   # → paper/manuscript.pdf
```

---

## Research questions (from project synopsis)

- **RQ1:** Are GNNs more vulnerable to MIAs than non-graph baselines (MLP / logistic regression)?
- **RQ2:** How do graph structure (homophily, density, sparsity) affect leakage?
- **RQ3:** Can lightweight defenses reduce leakage with minimal utility loss?
- **RQ4:** What is the utility–privacy tradeoff frontier?

---

## Deliverables (all in this repo)

- Reproducible, **config-driven** codebase  
- Paper experiment grid with Cora, Citeseer, six synthetic regimes, four model families, five random seeds, and confidence/threshold/shadow attacks  
- Publication figures: Attack AUC vs model, utility–privacy tradeoff, leakage vs homophily/density, defense effectiveness, and heatmaps  
- IEEE-style manuscript source in `paper/ieee_privacy_gnn.tex`  
- **Significance testing** (defense vs no defense) in `results/significance.csv`  

---

## Citation datasets (Cora, Citeseer)

Cora and Citeseer are downloaded via PyTorch Geometric (Planetoid). If the default download fails (e.g. network or GitHub raw access), the code tries a mirror automatically (`data.py`). If both fail, experiments still run for synthetic datasets only; re-run with working network to include citation results. Figure generation works with or without citation data (Fig 1 shows "No data" for missing Cora/Citeseer; Fig 2 uses a synthetic dataset as fallback).
