# HARP: A Selective Release Framework for Privacy-Audited GNN Prediction APIs

Artifact for the IEEE BigData 2026 submission.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export PRIVACYGNN_CONFIG=experiment_config_confirmatory.yaml
python run_harp_baselines.py              # primary grid
python run_harp_competitiveness_upgrade.py  # MemGuard, masking, audit seeds
python run_harp_eval.py         # shadow/cache/hop/ε/failure/ogbn LiRA
```

## Key modules

| Path | Role |
|------|------|
| `defenses/harp.py` | LTE / hop expand / Mass–Frac scales |
| `defenses/memguard.py` | Simplified MemGuard release baseline |
| `defenses/lbp.py` | Uniform Laplace posterior noise |
| `experiment.py` | Train + defense-aware LiRA / ECE |
| `run_harp_eval.py` | Shadow scaling, cache sim, hop necessity, local ε, failure cases, ogbn/products LiRA |
| `run_harp_*.py` | Paper tables and ablations |
| `paper/ieee_privacy_gnn.tex` | Manuscript |

## Locked HARP

`LOCKED_HARP` in `defenses/harp.py`: Frac=0.40, σ_strong=0.30, k=1, λ=0.5.

## Citation

See `paper/ieee_privacy_gnn.tex`.
