# Reproduce the HARP IEEE BigData 2026 experiments

Artifact for: *HARP: ExactFrac-Constrained Score Serving for GNN Prediction APIs*.

## Environment

```bash
cd privacy-gnn
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
export PRIVACYGNN_CONFIG=experiment_config_confirmatory.yaml
```

Prefer `./venv/bin/python` if a project venv already exists.

## Headline submission tables (recommended order)

```bash
# 1) Cora headline Acc/LiRA/ECE/ExactFrac at n_shadows=16 (5 seeds)
./venv/bin/python run_nsh16_headline.py

# 2) CFS under ExactFrac≥0.60 and τ=0.70 at n_shadows=16
./venv/bin/python run_cfs_nsh16.py

# 3) ExactFrac SLA evidence (sticky ≠ ExactFrac) + constructor/DCS/GAP/serving
./venv/bin/python run_exactfrac_sla_evidence.py
./venv/bin/python run_bulletproof.py
./venv/bin/python run_stack_slice.py

# 4) Canary-stressed ogbn-products BFS-15k probe
./venv/bin/python run_products_canary_leakage.py

# 5) Multi-dataset Acc/LiRA grid (n_shadows=4) + legacy sweeps
./venv/bin/python run_harp_baselines.py
./venv/bin/python run_harp_eval.py
./venv/bin/python run_harp_ogbn.py
```

Refresh paper figures from frozen CSVs:

```bash
./venv/bin/python make_submission_figures.py
```

## Frozen CSVs ↔ paper tables

| Paper table / figure | Primary CSV / figure |
|----------------------|----------------------|
| Table fair / Fig nsh16 | `results/harp_headline_nsh16(_means).csv`, `fig_harp_nsh16_scatter.png` |
| Table CFS | `results/harp_cfs_nsh16.csv`, `harp_cfs_nsh16_grid.csv` |
| Table constructors / slice | `results/harp_constructor_slice.csv`, `harp_stack_slice.csv`, `harp_dcs_slice.csv` |
| Table harp (6-graph, n_sh=4) | `results/harp_means.csv`, `harp_baselines.csv` |
| Table frac / session | `results/harp_frac_sweep_5seed.csv`, `harp_session_b_sweep_cora.csv` |
| Table cache / serving | `results/harp_replay_flicker.csv`, `harp_serving_v2.csv`, `harp_serving_bench.csv` |
| Table ExactFrac SLA (`tab:sla`) | `results/harp_exactfrac_sla_evidence.csv` (RawExactFrac vs ReplayFrac; includes global-cache + seeded LBP) |
| Table products / canary | `results/harp_products_canary_leakage.csv`, `harp_products_sub*_means.csv` |
| Table ogbn-arxiv | `results/harp_ogbn.csv`, `harp_ogbn_lira4_means.csv` |
| HARP∘GAP | `results/harp_gap_composition.csv` |
| Fig headline / cache | `figures/fig_harp_headline.png`, `fig_harp_cache_veracity.png` |

## Locked HARP

`LOCKED_HARP_RELEASE` in `defenses/harp.py`: Frac=0.40, σ_strong=0.30, k=1, release-only.

- Headline LiRA: `n_shadows=16`
- Multi-dataset / Acc dials: `n_shadows=4`
- CFS under headline budget: `τ=0.70` (not 0.55)

Optional DCS: `deterministic_confidence_smooth` in `defenses/harp.py`.

## Manuscript

```bash
cd paper && pdflatex ieee_privacy_gnn.tex && pdflatex ieee_privacy_gnn.tex
```

Target: ≤10 pages (IEEE BigData regular).

## Smoke check

```bash
PRIVACYGNN_CONFIG=experiment_config_smoke.yaml ./venv/bin/python -c \
  "from defenses.harp import LOCKED_HARP_RELEASE; print(LOCKED_HARP_RELEASE)"
./venv/bin/python -m pytest tests/test_harp.py -q
```

## Note on older scripts

Scripts named `run_scml_*`, `run_sami_*`, `analyze_leakage_law.py`, and `build_spab_release.py` are historical predecessors and are **not** required for the HARP paper tables above.
