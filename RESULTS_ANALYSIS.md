# In-Depth Analysis of PrivacyGNN Results

This document interprets your experiment results (420 runs over 6 synthetic datasets, 4 models, 6 defense configurations, 5 seeds) in the context of your research questions and project goals.

---

## 1. What the Results Contain

- **Scope:** Synthetic graphs only (Cora/Citeseer were not in this run due to download issues). So all conclusions below are for **controlled synthetic settings** (homophily × density).
- **Metrics per run:** Test accuracy, F1, AUROC (utility); confidence-based attack AUC/acc, threshold attack AUC/acc, shadow-model attack AUC/acc (privacy); ECE on test set (calibration).
- **Aggregation:** Mean ± std over 5 seeds per (dataset, model, defense) in `summary.csv`.
- **Significance:** Paired t-tests (defense vs no defense, same seeds) in `significance.csv`.

---

## 2. RQ1: Are GNNs More Vulnerable Than Non-Graph Baselines?

**Short answer: On your synthetic data, it depends on graph structure. When the task is easy (high homophily, enough density), no model leaks much. When the task is hard (low homophily, sparse), GCN leaks the most; GraphSAGE and baselines can look similar or better.**

### By dataset (no defense, confidence attack AUC)

| Dataset                  | LogReg | MLP   | GCN     | GraphSAGE |
|--------------------------|--------|-------|---------|-----------|
| synthetic_high_dense     | 0.485  | 0.512 | **0.494** | 0.509   |
| synthetic_high_medium    | 0.526  | 0.515 | **0.487** | 0.526   |
| synthetic_high_sparse    | 0.532  | 0.535 | **0.528** | 0.539   |
| synthetic_low_dense      | 0.479  | 0.517 | **0.511** | **0.510** |
| synthetic_low_medium     | 0.535  | 0.547 | **0.563** | 0.534   |
| synthetic_low_sparse     | 0.508  | 0.519 | **0.592** | **0.520** |

- **High homophily (all densities):** Attack AUC is in the 0.49–0.54 range for every model—close to random (0.5). So on “easy” graphs where everyone generalizes well, **GNNs are not more vulnerable**; no model leaks much.
- **Low homophily:** GCN’s attack AUC rises to **0.56 (medium)** and **0.59 (sparse)** while LogReg/MLP/GraphSAGE stay near 0.51–0.53. So **GCN is more vulnerable** when the graph is low-homophily and sparse; GraphSAGE stays close to baselines.

**Interpretation for your project:**  
- On “easy” synthetic graphs (high homophily), you do **not** see GNNs leaking more than MLP/LogReg; leakage is low for all.  
- On “hard” graphs (low homophily, especially sparse), **GCN shows the highest membership leakage**; GraphSAGE behaves more like the baselines. So vulnerability is **model- and structure-dependent**, not uniformly “GNNs worse than baselines.”

---

## 3. RQ2: How Do Homophily and Density Affect Leakage?

**Short answer: Low homophily and sparsity increase leakage for GCN; for GraphSAGE and baselines the effect is smaller. When models generalize almost perfectly (high homophily, dense), leakage stays near 0.5.**

### Homophily

- **High homophily (~0.80):** Across sparse/medium/dense, confidence attack AUC stays in the **0.49–0.54** band for all models. Little exploitable gap between train and test.
- **Low homophily (~0.30):**  
  - **GCN:** AUC grows from ~0.51 (low_dense) → ~0.56 (low_medium) → **~0.59 (low_sparse)**.  
  - **GraphSAGE:** Stays around 0.51–0.53 except label_smoothing on low_sparse (see defenses).  
  - **LogReg/MLP:** Remain ~0.48–0.55.

So **leakage increases as homophily decreases** for GCN; GraphSAGE and baselines are less sensitive.

### Density (within same homophily)

- **High homophily:** Sparse gives slightly higher AUC (e.g. GCN 0.528, GraphSAGE 0.539) than dense (0.494, 0.509). So **sparsity slightly increases leakage** when homophily is high.
- **Low homophily:** The main effect is homophily; density adds a smaller modulation (sparse is still worst for GCN).

**Interpretation:**  
- **Structure matters most when the model struggles to generalize** (e.g. GCN on low-homophily sparse graphs).  
- Your results support: “MIA vulnerability is tied to **generalization gap**”; when accuracy is near 1.0 and ECE is near 0, attack AUC is near 0.5 regardless of density/homophily.

---

## 4. RQ3: Do Lightweight Defenses Reduce Leakage?

**Short answer: No defense consistently reduces attack AUC across all settings. Label smoothing often increases leakage and hurts calibration; edge sparsification sometimes helps (e.g. GCN low_medium). DropEdge and early stopping are mixed; confidence masking is mostly neutral.**

### Summary by defense (confidence attack AUC, vs no defense)

- **Label smoothing:**  
  - **Increases** attack AUC in several cases (e.g. GraphSAGE synthetic_high_sparse: 0.539 → **0.577**; GraphSAGE synthetic_low_sparse: 0.520 → **0.614**; GraphSAGE synthetic_low_medium: 0.534 → **0.559**).  
  - ECE jumps (e.g. 0.00 → ~0.09–0.10), so predictions become more overconfident in a way the attacker can use.  
  - **Conclusion:** Label smoothing is **not** a privacy defense here; it often **increases** leakage.

- **Edge sparsification:**  
  - **GCN synthetic_low_medium:** AUC 0.563 → **0.538** (reduction ~0.025), and in significance testing this is one of the few **significant** effects (p ≈ 0.0015).  
  - Other settings: small or no improvement, sometimes slight increase.  
  - **Conclusion:** Modest, **setting-dependent** benefit; best visible for GCN on low-homophily medium-density graphs.

- **DropEdge:**  
  - **GCN synthetic_low_medium:** AUC 0.563 → 0.540 (p ≈ 0.055).  
  - Elsewhere: small or positive changes in AUC (e.g. GCN high_sparse 0.528 → 0.537).  
  - **Conclusion:** Slight improvement only in some low-homophily settings; can worsen leakage elsewhere.

- **Early stopping:**  
  - Often almost identical to no defense (same AUC, diff = 0 in several rows).  
  - **Conclusion:** No meaningful privacy gain in this setup.

- **Confidence masking:**  
  - Small ups and downs; no clear pattern.  
  - **Conclusion:** Largely **neutral** for attack AUC.

### Significance (significance.csv)

- **Statistically significant (p < 0.05) effects:**  
  - **GCN, synthetic_low_medium, edge_sparsification:** AUC **decrease** (diff ≈ -0.025, p ≈ 0.0015)—defense **helps**.  
  - **GraphSAGE, synthetic_low_sparse, label_smoothing:** AUC **increase** (diff ≈ +0.094, p ≈ 0.042)—defense **hurts**.  
  - **GraphSAGE, synthetic_low_sparse, edge_sparsification:** AUC decrease (diff ≈ -0.037, p ≈ 0.073)—marginal.

So the only clear, significant **privacy improvement** in your table is **edge sparsification for GCN on low-homophily medium graphs**. The only clear **privacy harm** is **label smoothing for GraphSAGE on low-homophily sparse graphs**.

---

## 5. RQ4: Utility–Privacy Tradeoff Frontier

**Short answer: When the model generalizes well (high accuracy, low ECE), you sit near the “good” corner (high utility, low leakage). When GCN underperforms (e.g. low_sparse), you get the worst tradeoff. Among defenses, edge sparsification gives a favorable trade in the settings where it reduces AUC without large accuracy loss.**

- **Best tradeoff regions:**  
  - High homophily + dense/medium: All models have test accuracy ~0.99–1.0 and attack AUC ~0.49–0.53—**high utility, low leakage**.  
  - GraphSAGE (no defense or most defenses) on low_homophily: Often 1.0 accuracy and ~0.51–0.53 AUC—again good tradeoff.

- **Worst tradeoff:**  
  - **GCN on synthetic_low_sparse:** Accuracy ~0.75–0.78, attack AUC ~0.59—**lower utility and higher leakage**.  
  - Adding label smoothing for GraphSAGE on low_sparse: AUC jumps to ~0.61 with no accuracy gain—**worse privacy for same utility**.

- **Defense tradeoff:**  
  - **Edge sparsification** on GCN low_medium: AUC drops (better privacy) with limited accuracy loss—**Pareto improvement**.  
  - **Label smoothing:** Often **Pareto worse** (higher AUC, similar or worse calibration).

So the **utility–privacy frontier** in your results is dominated by (1) **model and graph structure** (GCN + low homophily + sparse = bad), and (2) **avoiding label smoothing** and considering **edge sparsification** where it helps.

---

## 6. Shadow-Model vs Confidence-Based Attack

- **Shadow attack AUC** is often **~0.50** (random) when the confidence-based attack is also near 0.5 (e.g. high_homophily dense).  
- Where the **confidence attack** is high (e.g. GCN low_sparse ~0.59), **shadow attack** is also elevated (e.g. ~0.61).  
- So the shadow-model attack **corroborates** the confidence-based story: when there is a train–test confidence gap, both attack types exploit it; when there isn’t, both are at chance.

---

## 7. Calibration (ECE)

- **ECE** is near **0** when accuracy is near 1.0 and no label smoothing (e.g. GraphSAGE on most settings, LogReg/MLP on all).  
- **Label smoothing** consistently **increases ECE** (e.g. 0.09–0.11 for GCN/GraphSAGE with label smoothing).  
- **GCN on low_homophily** (especially sparse) has higher ECE (~0.05–0.10) even without label smoothing, matching worse calibration when the model struggles.

So **calibration error** aligns with **privacy leakage**: higher ECE tends to go with higher attack AUC.

---

## 8. What This Means for Your Research Project

1. **RQ1 (GNN vs baselines):** On synthetic data, GNNs are **not** uniformly more vulnerable. **GCN** is more vulnerable when the graph is **low-homophily and sparse**; GraphSAGE and baselines are similar or better. So your answer is: “It depends on model and structure; GCN can be worse in adverse structure.”

2. **RQ2 (Structure):** **Low homophily and sparsity** increase leakage, especially for GCN. High homophily and higher density keep leakage near random. This supports the “generalization gap drives leakage” narrative.

3. **RQ3 (Defenses):**  
   - **Do not** recommend **label smoothing** as a privacy defense; it often **increases** leakage and ECE.  
   - **Edge sparsification** can **help** in specific settings (GCN, low homophily, medium density) and is the only defense with a significant privacy gain in your tests.  
   - DropEdge, early stopping, and confidence masking do not show a consistent privacy benefit.

4. **RQ4 (Tradeoff):** Best utility–privacy tradeoffs occur when the model generalizes well (high homophily or robust model like GraphSAGE). Worst is GCN on low-homophily sparse graphs. Edge sparsification can improve the frontier where it reduces AUC; label smoothing worsens it.

5. **Limitation:** All of this is on **synthetic** graphs. Cora/Citeseer (and any real citation or healthcare data) should be run when possible to state conclusions for real-world settings and to compare with prior work (e.g. “GNNs leak less than MLP on Cora”).

---

## 9. Suggested Paper/Poster Takeaways

- **Headline:** On controlled synthetic graphs, **GCN is more vulnerable to MIAs when graph structure is unfavorable** (low homophily, sparse); **GraphSAGE and non-graph baselines are less sensitive**.  
- **Defenses:** **Label smoothing increases membership leakage**; **edge sparsification** can reduce it in some settings and is the only defense with a statistically significant privacy improvement in your study.  
- **Mechanism:** Vulnerability tracks **generalization and calibration**: when accuracy is high and ECE low, attack AUC is near 0.5; when the model struggles (e.g. GCN on low-homophily sparse), both calibration and leakage worsen.  
- **Reproducibility:** You provide config-driven experiments, 5 seeds, mean ± std, and significance tests—this supports a “systematic benchmarking” contribution for GNN privacy evaluation.
