# Literature Review: Privacy Leakage and Lightweight Defenses in Graph Neural Networks

## 1. Membership Inference Attacks on Machine Learning Models

Membership inference attacks (MIAs) aim to determine whether a specific data point was used in training a machine learning model. The foundational work by Shokri et al. (2017) introduced the shadow model approach, where an adversary trains multiple surrogate models to learn the distinction between members and non-members based on prediction confidence vectors. This approach has since been extended and refined across various domains.

Recent advances include confidence-based attacks that exploit prediction entropy without requiring shadow models (Yeom et al., 2018), metric-based attacks using loss thresholding (Carlini et al., 2022), and quantile-regression-based approaches for scalability (Bertran et al., 2023). The RMIA framework (Zarifzadeh et al., 2024) introduced robust likelihood ratio tests with low computational overhead.

## 2. MIAs Specific to Graph Neural Networks

The translation of MIA to graph-structured data introduces unique challenges due to the relational nature of graphs. Key works include:

- **He et al. (2021)** performed the first comprehensive analysis of node-level MIA against GNNs, demonstrating vulnerability even with minimal adversary background knowledge. They showed graph density and feature similarity significantly impact attack success.

- **Olatunji et al. (2021)** introduced two realistic MIA settings for GNNs and critically showed that **structural information is the major contributing factor** for privacy leakage in GNNs, beyond mere overfitting. They proposed two defenses achieving up to 60% reduction in attacker inference.

- **Guan et al. (2025)** proposed topology-based node-level MIA using neighbor information with effective feature processing strategies for variable-length features. They introduced multiple shadow model training and random non-membership selection strategies.

- **Wang & Wang (2024)** introduced Structure Membership Inference Attacks (SMIA) targeting subgraph-level membership, extending beyond node and edge-level attacks.

- **Jnaini et al. (2022, 2025)** evaluated MIA power on GNNs and proposed defense methods including LLM-guided posterior-level defenses.

## 3. Role of Graph Structure in Privacy Leakage

A critical finding across the literature is that **graph structure amplifies privacy leakage**:

- **Yuan et al. (2024)** introduced Graph Privacy Leakage via Structure (GPS) and proposed the Generalized Homophily Ratio to quantify privacy breach risks. This directly connects homophily to vulnerability.

- **Mueller et al. (2023)** empirically investigated differentially private GNNs on medical population graphs, finding a **correlation between the degree of graph homophily and model accuracy**, with implications for privacy-utility tradeoffs.

- He et al. (2021) demonstrated that graph density has a major impact on MIA success rates.

- The transductive learning setting common in GNNs (where test nodes are visible during training) creates additional leakage vectors (Niu et al., 2024).

**Gap identified:** While individual studies have noted the role of homophily and density, no systematic benchmarking study has controlled for these factors across multiple GNN architectures with a comprehensive defense evaluation.

## 4. Defense Mechanisms

### 4.1 Heavyweight Defenses (Differential Privacy)
- **PrivGNN** (Olatunji et al., 2023): Full DP framework for GNNs with formal privacy guarantees.
- **TDP-GNN** (Lei et al., 2025): Topology-aware DP with personalized privacy budgets per node.
- **ProGAP** (Sajadmanesh & Gatica-Perez, 2023): Progressive DP-GNNs balancing accuracy and privacy.
- **NFDP** (Chen et al., 2025): Noise-adaptive DP with adaptive clipping.

### 4.2 Lightweight/Empirical Defenses
- **DropEdge** (Rong et al., 2020): Random edge removal during training, originally for over-smoothing prevention but with privacy implications.
- **Graph perturbation** (Wu et al., 2022): Defense through controlled graph structure perturbation with visible privacy-utility tradeoffs.
- **PriGraph** (Shen et al., 2025/2026): Adversarial perturbation-based defense with minimal utility loss.
- **Qi et al. (2025)**: Training-inference collaborative framework using high-entropy soft labeling and entropy regularization.
- **Confidence masking**: Truncating or perturbing output posteriors at inference time.
- **Label smoothing**: Softening training targets to reduce overfitting-driven memorization.
- **Early stopping**: Halting training before overfitting, reducing the train-test generalization gap that MIAs exploit.

### 4.3 Critical Evaluation of Empirical Defenses
**Aerni et al. (2024)** presented a crucial finding: empirical privacy evaluations are often **misleading** — they underestimate privacy leakage by an order of magnitude when using weak attacks and fail to compare with properly tuned DP-SGD baselines. This motivates our use of both confidence-based AND shadow-model attacks for thorough evaluation.

## 5. Research Gaps and Our Contributions

Based on this review, we identify the following gaps that our work addresses:

1. **No systematic graph-vs-non-graph comparison**: While individual papers compare GNN vulnerability, no unified benchmark compares MLP/LogReg baselines against GCN/GraphSAGE under identical experimental conditions with statistical rigor.

2. **Structural factors underexplored**: Homophily and density effects on MIA are noted qualitatively but not systematically quantified with controlled synthetic graphs.

3. **Lightweight defense benchmarking absent**: Most defense papers propose a single method. No comprehensive evaluation compares DropEdge, label smoothing, early stopping, confidence masking, and edge sparsification head-to-head on the same datasets.

4. **Privacy-utility Pareto frontier not characterized**: While tradeoff curves are mentioned conceptually, no paper provides a thorough Pareto analysis of lightweight defenses across graph structural conditions.

Our work fills these gaps through a systematic benchmarking and empirical analysis approach, providing a reproducible experimental framework for GNN privacy evaluation.

## Key References

1. He, X., et al. (2021). "Node-Level Membership Inference Attacks Against Graph Neural Networks." arXiv:2102.05429.
2. Olatunji, I.E., et al. (2021). "Membership Inference Attack on Graph Neural Networks." IEEE TPSISA.
3. Yuan, H., et al. (2024). "Unveiling Privacy Vulnerabilities: Investigating the Role of Structure in Graph Data." arXiv:2407.18564.
4. Mueller, T.T., et al. (2023). "Privacy-Utility Trade-offs in Neural Networks for Medical Population Graphs." arXiv:2307.06760.
5. Guan, F., et al. (2025). "Topology-Based Node-Level MIA on GNNs." IEEE TBDATA.
6. Aerni, M., et al. (2024). "Evaluations of ML Privacy Defenses are Misleading." ACM CCS.
7. Qi, F., et al. (2025). "Training-Inference Collaborative Defense Against MIA for GNNs." SCIENGINE.
8. Wu, J., et al. (2022). "Defense against MIA in GNNs through Graph Perturbation." Int J Info Security.
9. Shen, M., et al. (2025). "PriGraph: Defending Against Inference Attacks on GNNs." IEEE TDSC.
10. Niu, P., et al. (2024). "Graph Transductive Defense." arXiv:2406.07917.
