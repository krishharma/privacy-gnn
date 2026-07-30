# Evaluation Protocol (HARP, pre-declared)

HARP evaluates **constrained score release** for GNN prediction APIs:
maximize Acc subject to a membership audit and ExactFrac ≥ *c*.
HARP is **not** proposed as a stronger membership defense than uniform noise when ExactFrac=0 is allowed.

## Primary metrics

| Role | Metric |
|------|--------|
| Primary utility | Test accuracy |
| Membership audit | Defense-aware LiRA AUROC; TPR @ FPR ∈ {0.001, 0.01} |
| Data quality | ExactFrac (= 1 − Frac), ECE, clean-slice ECE |
| Systems spend | Mass (= Σ σ_v), Frac, session budget *B*, train/release wall time |

LiRA tells operators *whether* to spend noise; Acc and ECE tell them *what they paid*.

## Splits and seeds

- Citation / heterophilic / co-purchase: **40% / 20% / 40%** train/val/test, resampled per seed.
- ogbn-arxiv: official OGB split for the target; NeighborLoader training.
- ogbn-products LiRA: BFS-induced 15k-node subgraph with 40/20/40 resplit (full-graph LiRA exceeds 16 GB host RAM).
- Members = train-mask nodes; non-members = test-mask nodes.
- Hyperparameters (Frac, σ, *k*, warmup) selected on **validation Acc** — never to minimize test LiRA.
- Cora fairness / Frac: 5 seeds; other cells: 3 seeds `{42,123,456}` unless noted.

## LiRA shadow counts

| Regime | *n*_shadows | Role |
|--------|-------------|------|
| Citation / Chameleon / Actor (primary) | **4** | Defense-aware LiRA under CPU budget |
| Cora / Chameleon stability | **4, 16, 64** | Shadow-count comparative table |
| ogbn-arxiv primary | **2** | Cost-gated volume Acc recovery |
| ogbn-arxiv stress | **4** | Confirm audit-null under more shadows |
| products BFS subsample | **2** | Honest large-graph audit proxy |

## Locked HARP and baselines

Locked: Frac=0.40, σ_strong=0.30, *k*=1, LTE constructor, Laplace protector.

Baselines:

- Strong LBP (*b*=0.3) — published Acc-collapse recipe
- Equal-mass LBP (*b*≈0.12) — matched-Mass fairness when ExactFrac=0 is allowed
- Simplified MemGuard — every response perturbed (ExactFrac=0)
- Selective masking — ExactFrac>0 protector ablation
- GTD / MaskArmor — train-time defenses (no Mass/Frac)
- Clip+noise DP-SGD — formal-privacy anchor (vacuous ε at Acc-tuned point)

## Success criteria (pre-declared)

1. **ExactFrac feasibility:** HARP achieves ExactFrac = 1 − Frac; uniform LBP cannot for *c*>0.
2. **Acc recovery:** HARP Acc ≫ strong LBP Acc at locked Frac on leaky and volume cells.
3. **Matched-Mass honesty:** equal-mass LBP may beat HARP on Acc–LiRA when ExactFrac=0; report this.
4. **Audit-first spend:** if undefended LiRA ≈ 0.5, recommend Frac=0; Acc recovery under audit-null is a systems demo, not a privacy claim.
5. **Shadow honesty:** report none/LBP/HARP LiRA as *n*_shadows grows; do not claim HARP beats LBP on LiRA at large *n*_shadows.

## Reproduce entrypoints

See `REPRODUCE.md`:

```bash
./venv/bin/python run_harp_baselines.py
./venv/bin/python run_harp_competitiveness_upgrade.py
./venv/bin/python run_harp_eval.py
./venv/bin/python run_harp_ogbn.py
```

## What this protocol is not

- Not a claim of differential privacy (use GAP for legal ε).
- Not a claim that clean-majority nodes are membership-safe.
- Not a requirement to re-run historical SAMI/SCML/SPAB scripts for the HARP paper.
