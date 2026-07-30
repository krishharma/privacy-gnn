# PrivacyGNN: SPAB + SCML + SAMI

**PI:** Dr. Shaikh Arifuzzaman  
**Student Researcher:** Krish Sharma  

**SPAB** — Structure-conditioned Privacy Audit Benchmark at Volume (ogbn-arxiv NeighborLoader systems metrics).  
**SCML** — Structure-Conditioned Membership Leakage regularity (held-out synthetic LOO + feature-SNR axis).  
**SAMI** — architecture-aware LTE + risk-weighted φ-alignment (minimal intervention).

## Quick start

```bash
cd privacy-gnn
./venv/bin/python run_core_tables.py
./venv/bin/python run_scml_expanded.py
./venv/bin/python run_mia_eval_standard.py
./venv/bin/python run_sami_budget_protocol.py
PRIVACYGNN_CONFIG=experiment_config_ogbn_smoke.yaml ./venv/bin/python run_ogbn_smoke.py
./venv/bin/python run_ogbn_volume.py
./venv/bin/python freeze_paper_release.py
```

See [`REPRODUCE.md`](REPRODUCE.md), [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md), `paper/ieee_privacy_gnn.pdf`, `results/paper_release/`.

## Headline results (frozen)

- **Cora GraphSAGE:** SAMI beats GTD on conf / LiRA / TPR@1%FPR (WIN framing in `sami_gtd_framing.json`).
- **SCML LOO:** Spearman ≈0.70, MAE ≈0.038 (`leakage_law_fit.json`).
- **ogbn-arxiv:** 169k-node Volume audit; near-chance undefended MIA; systems QPS ≈27k (`ogbn_volume_results.csv`).
- **GCN hard cell:** still mixed after arch-aware LTE (`gcn_hardcell_best.json`).
