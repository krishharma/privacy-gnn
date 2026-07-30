"""Freeze results/paper_release/ with config hash, tables, timing, and protocol notes."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

import pandas as pd

from config import load_config
from stats_utils import (
    run_bootstrap_summary,
    run_bootstrap_delta_summary,
    run_confirmatory_tests,
    run_significance_tests,
    run_significance_tests_lira,
)
from summarize_paper_tables import main as summarize_main


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(root, "results")
    snap = os.path.join(res, "paper_release")
    os.makedirs(snap, exist_ok=True)

    os.environ.setdefault("PRIVACYGNN_CONFIG", "experiment_config_confirmatory.yaml")
    cfg = load_config()

    src = os.path.join(res, "core_results.csv")
    if not os.path.isfile(src):
        src = os.path.join(res, "all_results.csv")
    df = pd.read_csv(src)
    df.to_csv(os.path.join(snap, "all_results.csv"), index=False)
    df.to_csv(os.path.join(res, "all_results.csv"), index=False)

    summarize_main()
    run_significance_tests(df, os.path.join(res, "significance.csv"))
    run_significance_tests_lira(df, os.path.join(res, "significance_lira.csv"))
    run_confirmatory_tests(df, os.path.join(res, "significance_confirmatory.csv"))
    run_bootstrap_summary(df, os.path.join(res, "summary_bootstrap.csv"))
    run_bootstrap_delta_summary(df, os.path.join(res, "summary_delta_bootstrap.csv"))

    copy_names = [
        "paper_tables_summary.csv",
        "significance.csv",
        "significance_lira.csv",
        "significance_confirmatory.csv",
        "summary_bootstrap.csv",
        "summary_delta_bootstrap.csv",
        "leakage_law_fit.json",
        "leakage_law_train.csv",
        "leakage_law_oos.csv",
        "leakage_law_loo.csv",
        "leakage_law_intervention.csv",
        "architecture_modulation.csv",
        "feature_reversal.csv",
        "feature_snr_grid.csv",
        "scml_expanded_raw.csv",
        "gcn_hardcell_best.json",
        "sami_gtd_framing.json",
        "sami_vs_gtd_joint.csv",
        "maskarmor_5seed.csv",
        "gap_attack_table.csv",
        "lte_quintile_tpr.csv",
        "lte_quintile_attribution.csv",
        "lte_quintile_attribution_summary.json",
        "lte_phi_gap_spearman.csv",
        "lte_mechanism_audit.json",
        "mia_eval_standard.json",
        "ogbn_smoke_timing.json",
        "ogbn_volume_results.csv",
        "ogbn_systems.json",
        "ogbn_lira_n4_oneseed.json",
        "ogbn_lira_n4_3seed.csv",
        "ogbn_lira_n4_3seed_summary.json",
        "planetoid_cora_appendix.csv",
        "planetoid_cora_appendix_summary.json",
        "citeseer_retune_confirm.csv",
        "citeseer_retune_summary.json",
        "chameleon_baselines.csv",
        "chameleon_baselines_summary.json",
        "actor_baselines.csv",
        "actor_baselines_summary.json",
        "systems_audit_cost.csv",
        "spab_report.csv",
        "spab_schema.json",
        "cora_lira_n8.csv",
        "cora_lira_n8_summary.json",
        "volume_highrisk_synth.csv",
        "volume_highrisk_synth_summary.json",
        "ogbn_gtd_volume_note.json",
        "sami_budget_confirm.csv",
        "sami_budget_select.csv",
        "actor_summary.json",
        "volume_variety_probe.json",
        "baselines_extra.csv",
        "dp_reference.csv",
        "timing_overhead.csv",
        "scaling_limitations.json",
        "power_analysis.json",
    ]
    for name in copy_names:
        p = os.path.join(res, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(snap, name))

    meta = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": os.environ.get("PRIVACYGNN_CONFIG", ""),
        "config_hash": cfg.get("config_hash", ""),
        "core_results_sha256_16": _hash_file(src),
        "n_rows": int(len(df)),
        "datasets": sorted(df["dataset"].astype(str).unique().tolist()),
        "defenses": sorted(df["defense"].astype(str).unique().tolist()),
        "reproduce": [
            "./venv/bin/python run_core_tables.py",
            "./venv/bin/python run_scml_expanded.py",
            "./venv/bin/python run_mia_eval_standard.py",
            "./venv/bin/python run_sami_budget_protocol.py",
            "./venv/bin/python run_ogbn_smoke.py",
            "./venv/bin/python run_ogbn_volume.py",
            "./venv/bin/python run_highrisk_volume_synth.py",
            "./venv/bin/python build_spab_release.py",
            "./venv/bin/python freeze_paper_release.py",
        ],
        "notes": {
            "primary_claim": "SPAB conditional audit (primary) + supporting regularity + one defense (SAMI).",
            "gcn_hardcell": "See gcn_hardcell_best.json (honest mixed / architecture modulation).",
            "ogbn_arxiv": "Volume negative control + systems; Volume-safe GTD Acc≈none.",
            "volume_x_leakage": "n=3k low-h/low-SNR GCN; see volume_highrisk_synth_summary.json.",
            "actor_variety": "5-seed high-risk cell; actor_summary.json.",
            "cora_lira_n8": "Stronger-shadow citation check; cora_lira_n8_summary.json.",
            "spab_report": "results/spab_report.csv matches paper 1:1.",
            "sami_vs_gtd": "WIN on Cora GraphSAGE joint metrics.",
            "defense_aware_shadows": True,
            "n_shadows": "Citation/Actor=4; Cora check=8; ogbn=2 (smoke wall-clock).",
        },
    }
    with open(os.path.join(snap, "RELEASE_META.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"Froze snapshot to {snap}")


if __name__ == "__main__":
    main()
