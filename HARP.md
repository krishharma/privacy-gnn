# HARP: Hop-Aware Risk-conditioned Privacy

## Quick start

```bash
cd privacy-gnn
./venv/bin/python -c "from defenses.harp import LOCKED_HARP; print(LOCKED_HARP)"
./venv/bin/python run_harp_baselines.py
./venv/bin/python make_harp_figures.py
./venv/bin/python scripts/fill_harp_table.py
```

## Algorithm

1. LTE risk $r_v$ from graph + train mask
2. Select top-risk seeds (or binary-search seed fraction to hit `target_protect_frac`)
3. Expand seeds by `k_hops` on the undirected graph
4. Laplace noise with `strong_noise_scale` only on the protected set
5. Optional: SAMI-style AdvReg with risk masked to the protected set

## Locked config

See `defenses/harp.py` → `LOCKED_HARP`.

## Metrics

`run_one` returns `noise_mass`, `frac_protected`, `frac_seeds`, `relative_noise_mass_vs_uniform`.
