# Reproduce the HARP IEEE BigData 2026 experiments

Artifact for: *HARP: Hop-Aware Selective Release for Privacy-Audited GNN Prediction APIs*.

## Environment

```bash
cd privacy-gnn
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Prefer `./venv/bin/python` if a project venv already exists.

```bash
export PRIVACYGNN_CONFIG=experiment_config_confirmatory.yaml
```

## Paper tables (recommended order)

```bash
# 1) Primary Acc / LiRA / Mass / Frac grid (six graphs)
./venv/bin/python run_harp_baselines.py

# 2) MemGuard, selective masking, audit/random seeds, slice ECE, session B, Frac=0.6
./venv/bin/python run_harp_competitiveness_upgrade.py

# 3) Shadow scaling, cache simulation, hop necessity, local ε,
#    Chameleon failure cases, ogbn-arxiv LiRA (n_shadows=4), products subsample LiRA
./venv/bin/python run_harp_eval.py

# 4) ogbn-arxiv Acc recovery (n_shadows=2 primary)
./venv/bin/python run_harp_ogbn.py
```

Optional extras used in the paper:

```bash
./venv/bin/python run_harp_adaptive_adversary.py
./venv/bin/python run_harp_shadow_sweep.py          # subset of run_harp_eval shadow block
./venv/bin/python run_ogbn_products_harp_systems.py  # full-graph systems probe (no LiRA)
```

## Frozen CSVs ↔ paper tables

| Paper table / figure | Primary CSV / figure |
|----------------------|----------------------|
| Table fair (Cora systems) | `results/harp_fairness_cora_5seed.csv`, `harp_memguard_mask_means.csv` |
| Table harp (6-graph grid) | `results/harp_means.csv`, `harp_baselines.csv` |
| Table eqmass / slice ECE | `results/harp_equal_mass_multids_means.csv`, `harp_slice_ece_multids_means.csv` |
| Table frac / session / mq | `results/harp_frac_sweep_5seed.csv`, `harp_session_b_sweep_cora.csv`, `harp_multi_query.csv` |
| Table ablate (constructors) | `results/harp_audit_seeds_means.csv`, `harp_lte_vs_uniform_delta.csv` |
| Table audit (shadow scaling) | `results/harp_shadow_comparative.csv`, `harp_shadow_sweep_means.csv` |
| Table cache | `results/harp_cache_simulation.csv` |
| Table canary / ogbn-arxiv | `results/harp_canary.csv`, `harp_ogbn.csv`, `harp_ogbn_lira4_means.csv` |
| Table products | `results/harp_products_sub_lira_means.csv`, `ogbn_products_harp_systems.json` |
| Fig hop / cache | `figures/fig_harp_hop_necessity.png`, `fig_harp_cache_veracity.png` |
| Failure-case narrative | `results/harp_failure_cases_chameleon.json` |

## Locked HARP

`LOCKED_HARP` in `defenses/harp.py`: Frac=0.40, σ_strong=0.30, k=1, λ=0.5.

LiRA primary: `n_shadows=4` (citation/heterophilic); ogbn-arxiv primary `n_shadows=2`, stress `n_shadows=4`.

## Manuscript

```bash
cd paper && pdflatex ieee_privacy_gnn.tex && pdflatex ieee_privacy_gnn.tex
```

## Smoke check

```bash
PRIVACYGNN_CONFIG=experiment_config_smoke.yaml ./venv/bin/python -c \
  "from defenses.harp import LOCKED_HARP; print(LOCKED_HARP)"
./venv/bin/python -m pytest tests/test_harp.py -q
```

## Note on older scripts

Scripts named `run_scml_*`, `run_sami_*`, `analyze_leakage_law.py`, and `build_spab_release.py` are historical predecessors. They are **not** required to reproduce the HARP paper tables above.
