# Evaluation Protocol (pre-declared, confirmatory)

SPAB = scale-aware **audit protocol + release CSV** (not a claim of community benchmark adoption).
Primary privacy = **LiRA**; conf AUROC is secondary.
Volume = NeighborLoader systems + **negative control** on ogbn-arxiv — not proof that large graphs leak.

## Primary metrics

| Role | Metric |
|------|--------|
| Primary privacy | LiRA attack AUROC; TPR @ FPR ∈ {0.001, 0.01} |
| Primary utility | Held-out (test) accuracy, macro-F1, macro AUROC |
| Secondary privacy | Confidence-attack AUROC, gap (label-only) AUROC, shadow AUROC, ECE |
| Systems (Volume / Velocity) | Train seconds, peak RSS, defended API QPS |

## Splits

- **40% train / 20% val / 40% test**, resampled per seed (citation/Actor/synthetics).
- **ogbn-arxiv (Volume negative control):** official OGB split for the target; shadows use resplit for membership diversity.
- **Actor (Variety-at-scale):** same 40/20/40 split; 5 confirmatory seeds; primary multi-thousand-node high-risk cell.
- Members for MIA = train-mask nodes; non-members = test-mask nodes.
- Validation is used for early stopping and as the SAMI defender’s non-member proxy. **Test nodes are never used by the defense.**
- Defense hyperparameters (incl. arch-aware LTE, risk budget B, hard-cell grid, ogbn batch/neighbors/epochs) are selected on **validation utility–privacy scores / Acc only** — **never** to minimize test attack AUROC. Confirmatory seeds `{42,123,456,789,1024}` are locked (Volume: `{42,123,456}`).

## LiRA \(n_{\mathrm{shadows}}\) (protocol, not attack-tuning)

| Regime | \(n_{\mathrm{shadows}}\) | Rationale |
|--------|--------------------------|-----------|
| Citation / Actor / synthetics (confirmatory) | **4** | Defense-aware LiRA under CPU budget |
| Cora GraphSAGE stronger-shadow check | **8** | Frozen in `cora_lira_n8_summary.json` (3 seeds) |
| ogbn-arxiv Volume | **2** | Smoke wall-clock gate (`ogbn_smoke_timing.json`) |
| Volume×leakage synth \(n{=}3\)k | **4** | High-risk imperfect-Acc cell |

Citation vs Volume shadow-count differences are a **systems** choice from measured wall-clock, **not** attack hyperparameter search.

## Volume (ogbn) + Variety-at-scale + Volume×leakage

```bash
PRIVACYGNN_CONFIG=experiment_config_ogbn_smoke.yaml ./venv/bin/python run_ogbn_smoke.py
./venv/bin/python run_ogbn_volume.py
./venv/bin/python run_ogbn_gtd_fix.py
./venv/bin/python run_highrisk_volume_synth.py
./venv/bin/python build_spab_release.py
```

- **ogbn official** = Volume negative control + systems.
- **Actor (5 seeds)** = real multi-k high-risk Variety.
- **High-risk synth** = controlled Volume×leakage (imperfect Acc).
- ogbn-products not cached (multi-GB); documented skip.

## SAMI vs GTD success bar + fallback

- **WIN:** on Cora GraphSAGE, SAMI beats GTD on ≥2 of {conf AUROC, LiRA AUROC, TPR@1% FPR} at matched-or-better Acc.
- **FALLBACK (pre-committed):** SAMI dominates the joint accuracy–LiRA frontier and does not inflate LiRA (unlike LBP); GTD may be marginally better on confidence alone but is structure-blind.
- **Volume GTD:** NeighborLoader uses Volume-safe stage-1-only CE (`stage1_frac≥0.99`). Unstable unlabeled pseudo-label stages that collapse Acc (~0.47) are discarded as reimplementation artifacts, not reported as GTD’s true utility.

## SCML (scientific primary)

```bash
./venv/bin/python run_scml_expanded.py
```

Primary claim uses **leave-one-regime-out** MAE/Spearman on synthetics with feature-SNR axis; demote in-sample R².
Wording: SCML is a **falsifiable predictive regularity**, not a “law,” and does not claim novelty for “structure affects MIA” alone.

## SPAB public report

```bash
./venv/bin/python build_spab_release.py
```

Fixed columns in `results/spab_report.csv`: regime, dataset, model, defense, \(h\), \(\rho\), \(n\), Acc, conf/LiRA AUROC, TPR@1%FPR, train_s, peak_MB, QPS, \(n_{\mathrm{shadows}}\), leakage_band.

## Volume×leakage high-risk synthetic

```bash
./venv/bin/python run_highrisk_volume_synth.py
```

GCN \(n{=}3000\), \(h{=}0.15\), sparse, SNR\(={0.05}\): imperfect Acc + high conf AUROC (fills Volume×leakage next to ogbn negative control).

## Stronger-shadow citation check

Cora GraphSAGE none/SAMI with \(n_{\mathrm{shadows}}{=}8\) (3 seeds): `results/cora_lira_n8_summary.json`.

## Out of scope (named)

RMIA, PPV-under-skewed-priors, injected canaries, inductive-split re-runs — awareness only; not claimed.

## Shared training defaults

- Optimizer: Adam, lr=0.01, weight decay=5e-4
- Epochs: 50 (citation); 20 NeighborLoader (ogbn); 80 (Volume×Variety synth)
- Device: CPU (paper Volume runs); document any GPU deviation

## Confirmatory comparisons

1. SAMI vs `none`
2. SAMI vs LBP
3. SAMI vs GTD
4. SAMI vs MaskArmor (5 seeds)
5. Ablations: full SAMI vs −LTE, −adv, −HCAG, temp-only, AdvReg (structure-blind)
6. Tuned DP-SGD Pareto anchor (not strawman Acc≈0.19)
7. Actor Variety stretch (5 seeds)
8. Volume negative control (ogbn official) + Actor Variety stretch (5 seeds)

Paired tests across seeds; **Holm–Bonferroni** within the confirmatory family; bootstrap CIs on paired ΔAUROC.

## Adaptive attackers

- Shadows for confidence / threshold / shadow / LiRA use the **same defense + release API** as the target (defense-aware by construction).
- Multi-query averaging against risk-scaled Laplace: K ∈ {1,5,20}.
- MLP-φ attacker on the same 4-D feature map alongside LR-φ.
- Label-only gap attack reported on headline cells.

## One-command reproduce

```bash
PRIVACYGNN_CONFIG=experiment_config_confirmatory.yaml ./venv/bin/python run_core_tables.py
./venv/bin/python run_scml_expanded.py
./venv/bin/python run_mia_eval_standard.py
./venv/bin/python run_sami_budget_protocol.py
PRIVACYGNN_CONFIG=experiment_config_ogbn_smoke.yaml ./venv/bin/python run_ogbn_smoke.py
./venv/bin/python run_ogbn_volume.py
./venv/bin/python run_ogbn_gtd_fix.py
./venv/bin/python run_highrisk_volume_synth.py
./venv/bin/python build_spab_release.py
./venv/bin/python freeze_paper_release.py
```

See `REPRODUCE.md` and `results/paper_release/`.
