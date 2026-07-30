# Zenodo / artifact deposit checklist (SPAB v1.0)

## Package contents
- `SPAB_v1_SPEC.md` — schema + seed/attack policy
- `results/spab_report.csv` + `spab_schema.json`
- `results/paper_release/` — frozen tables used in the paper
- `EVALUATION_PROTOCOL.md`, `REPRODUCE.md`
- Source runners listed in `RELEASE_META.json`

## Reproduce
```bash
./venv/bin/python build_spab_release.py
./venv/bin/python freeze_paper_release.py
# optional elevation:
./venv/bin/python run_ogbn_lira_n4.py
./venv/bin/python run_planetoid_appendix.py
./venv/bin/python run_citeseer_retune.py
./venv/bin/python run_lte_quintile_attribution.py
./venv/bin/python run_chameleon_baselines.py
./venv/bin/python run_systems_audit_cost.py
```

## DOI
Deposit package: `results/spab_v1_zenodo/spab_v1_*.zip` (see `DEPOSIT.md` there).

After publishing on Zenodo, run:
```bash
./venv/bin/python scripts/set_zenodo_doi.py 10.5281/zenodo.XXXXXXXX
```

Until then the DOI is pending deposit (no token in this environment).


## License
Same as repository (add LICENSE if missing before deposit).
