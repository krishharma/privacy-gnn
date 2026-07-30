# SPAB v1.0 — Scale-aware Privacy Audit Bundle
# Spec page for artifact release (Zenodo / paper supplement)

## Purpose
SPAB is an **audit protocol + release schema**, not a community leaderboard claim.
A SPAB row answers: under graph regime \((h,\rho,M,n)\) and split protocol \(P\),
what are Acc, LiRA AUROC, TPR@1%FPR, and systems cost for a defended prediction API.

## Column schema (`results/spab_report.csv`)
| Column | Type | Definition |
|--------|------|------------|
| `dataset` | str | Graph name |
| `model` | str | GCN / GraphSAGE |
| `defense` | str | none / sami / gtd / lbp / maskarmor / … |
| `homophily_h` | float | Edge homophily |
| `density_rho` | float | Edge density |
| `n_nodes` | int | Node count |
| `split_protocol` | str | `random_40_20_40` \| `planetoid_public` \| `ogb_official` \| `*_stress` |
| `acc` | float | Test accuracy |
| `lira_auroc` | float | Defense-aware Gaussian LiRA AUROC (**primary**) |
| `conf_auroc` | float | Confidence-attack AUROC (secondary) |
| `tpr_at_1pct_fpr` | float | LiRA TPR at 1% FPR |
| `train_seconds` | float | Target train wall (s) |
| `n_shadows` | int | LiRA shadow count for this row |
| `leakage_band` | str | Qualitative band for regime map |
| `notes` | str | Caveats (underpowered, stress, locked config, …) |
| `api_qps` | float\|null | Optional inference QPS |

## Seed & attack policy
- Citation / Actor / Chameleon: seeds `{42,123,456,789,1024}` unless noted.
- Volume (ogbn): 3 seeds; LiRA `n_shadows=2` in main grid; **`n_shadows=4` credibility CSV** for none+SAMI.
- Defense-aware shadows: same defense + release as target.
- Membership: train_mask = member; test_mask = non-member.

## Locked SAMI (paper tables)
```
lam=0.5, use_lte=True, use_gate=True, arch_aware=True,
noise_scale=0.35, budget_B=0.0, warmup_epochs=5, entropy_coef=0.05
```
ogbn SAMI uses `use_gate=False`, `noise_scale=0.1`, `warmup_epochs=3` (NeighborLoader config).

## Reproduce byte-identical CSV
```bash
./venv/bin/python build_spab_release.py
# → results/spab_report.csv + results/spab_schema.json
./venv/bin/python freeze_paper_release.py
```

## Version
- **SPAB v1.0** — IEEE BigData 2026 artifact companion
- Bump minor version when columns change; major when membership predicate changes.

## Citation
Cite the paper + this schema file. Zenodo DOI placeholder: replace after deposit.
