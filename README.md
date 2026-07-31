# HARP: ExactFrac-Constrained Score Serving for GNN Prediction APIs

Artifact for the IEEE BigData 2026 submission.

HARP is a **constrained score-serving framework**: keep ExactFrac ≥ c (replay-stable responses) by protecting a hop-expanded subset, with pluggable constructors and Constrained Frac Search (CFS). It is **not** a claim of universal membership-inference dominance or differential privacy.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export PRIVACYGNN_CONFIG=experiment_config_confirmatory.yaml
python run_harp_constrained_upgrade.py   # constructors, Spearman, serving, CFS
python run_harp_baselines.py             # primary Acc/LiRA grid
python run_harp_eval.py                  # cache/hop/ogbn probes
```

## Key modules

| Path | Role |
|------|------|
| `defenses/harp.py` | Constructors, hop expand, Mass–Frac, CFS, `LOCKED_HARP_RELEASE` |
| `defenses/memguard.py` | Simplified MemGuard release baseline |
| `defenses/lbp.py` | Uniform Laplace posterior noise |
| `experiment.py` | Train + defense-aware LiRA / ECE |
| `run_harp_constrained_upgrade.py` | Feasibility/constructors/serving upgrade |
| `paper/ieee_privacy_gnn.tex` | Manuscript |

## Locked config (paper default)

`LOCKED_HARP_RELEASE`: release-only, Frac=0.40 (ExactFrac=0.60), σ_strong=0.30, k=1, λ=0.

## Citation

See `paper/ieee_privacy_gnn.tex`.
