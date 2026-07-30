# Reproduce the IEEE BigData SCML + SPAB + SAMI experiments

## Environment

```bash
cd privacy-gnn
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Prefer `./venv/bin/python` if the project venv is already present.

## One-command confirmatory core (paper tables)

```bash
./venv/bin/python run_core_tables.py
# or full YAML grid:
PRIVACYGNN_CONFIG=experiment_config_confirmatory.yaml ./venv/bin/python run_final.py
```

Writes:
- `results/core_results.csv` / `results/all_results.csv` — per-seed rows (includes `config_hash`)
- `results/summary.csv` — mean/std over seeds (via `run_final.py`)
- `results/significance.csv` — paired tests vs `none` (conf AUROC) + Holm adjustment
- `results/significance_lira.csv` — same for LiRA AUROC
- `results/significance_confirmatory.csv` — SAMI vs baselines with effect sizes
- `results/summary_bootstrap.csv` — 95% bootstrap CIs over seeds
- `results/summary_delta_bootstrap.csv` — bootstrap CIs on paired **ΔAUROC**
- `results/power_analysis.json` — power paragraph for Cora GraphSAGE

```bash
./venv/bin/python run_stats_power.py
./venv/bin/python summarize_paper_tables.py
```


## Volume (ogbn-arxiv)

```bash
PRIVACYGNN_CONFIG=experiment_config_ogbn_smoke.yaml ./venv/bin/python run_ogbn_smoke.py
./venv/bin/python run_ogbn_volume.py
```

## SCML expanded + MIA-eval + SAMI budget

```bash
./venv/bin/python run_scml_expanded.py
./venv/bin/python run_mia_eval_standard.py
./venv/bin/python run_sami_budget_protocol.py
./venv/bin/python run_gcn_hardcell_arch.py
```

## Structure-Conditioned Membership Leakage (SCML)

Primary scientific analysis (fit on synthetics; architecture gap; feature reversal; intervention validity):

```bash
./venv/bin/python analyze_leakage_law.py
```

Writes `results/leakage_law_*.csv/json`, `figures/fig_leakage_law_pred.png`, `fig_intervention_validity.png`.

## Full 10-seed BigData matrix

```bash
PRIVACYGNN_CONFIG=experiment_config_bigdata.yaml ./venv/bin/python run_final.py
```

## Smoke test (wiring check)

```bash
PRIVACYGNN_CONFIG=experiment_config_smoke.yaml ./venv/bin/python run_final.py
```

## Figures

```bash
./venv/bin/python generate_bigdata_figures.py
./venv/bin/python run_pareto_acc_lira.py
```

## SAMI training / release (paper Algorithm pointer)

Matches Fig. SAMI and Eqs. (LTE + loss) in the paper:

1. Compute LTE risks \(r_v\) from degree / local heterophily / supervised-neighbor fraction; min–max to \([0,1]\).
2. For each epoch: forward GNN (+ optional HCAG); update discriminator \(D\) on detached \(\phi\) to separate train vs val; update GNN by CE + risk-weighted AdvReg/MMD + entropy.
3. Release \(\tilde p_v \leftarrow \mathrm{renorm}(\mathrm{softmax}(z_v/T_v)+\mathrm{Lap}(0,\sigma r_v))\).
4. Evaluate MIAs on train members vs test non-members only (same release on shadows).

Locked citation config: \(\lambda{=}0.5\), \(\sigma{=}0.35\), HCAG on. Implementation: `privacy/sami.py` / YAML defense blocks.

## Elevation experiments (canary LiRA / hop-off / DP Pareto)

```bash
./venv/bin/python run_canary_lira.py          # top-decile + planted canary LiRA
./venv/bin/python run_choquette_boundary.py   # Choquette-Choo-style hop-off
./venv/bin/python run_dp_pareto_tune.py       # Acc-tuned naive DP-SGD + LiRA
./venv/bin/python run_aerni_lte_worstcase.py  # conf top-decile + Spearman CIs
```

Writes `results/canary_lira_*.csv/json`, `choquette_boundary_*.csv/json`, `dp_pareto_*.csv/json`.

## Evaluation protocol

See [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) for pre-declared primary metrics,
40/20/40 splits, defense-aware shadows, adaptive attackers, confirmatory comparisons, and exploratory labels.

## Frozen release snapshot

```bash
./venv/bin/python freeze_paper_release.py
```

## Paper build

```bash
cd paper
pdflatex ieee_privacy_gnn.tex
pdflatex ieee_privacy_gnn.tex
pdflatex ieee_privacy_gnn.tex
```
