"""
Rebuild SPAB release: complete columns only; LiRA-primary regime map with split tags.
Schema drops peak_mb/api_qps unless measured for that row.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
PAPER_VIS = os.path.join(ROOT, "paper", "paper_visuals")
os.makedirs(FIG, exist_ok=True)
os.makedirs(PAPER_VIS, exist_ok=True)

# Columns that MUST be filled for every row (else omit row or use NaN only if noted).
REQUIRED = [
    "dataset",
    "model",
    "defense",
    "homophily_h",
    "density_rho",
    "n_nodes",
    "split_protocol",
    "acc",
    "lira_auroc",
    "conf_auroc",
    "tpr_at_1pct_fpr",
    "train_seconds",
    "n_shadows",
    "leakage_band",
]


def _m(df, dataset, model, defense, col):
    sub = df[(df.dataset == dataset) & (df.model == model) & (df.defense == defense)]
    if len(sub) == 0 or col not in sub.columns:
        return float("nan")
    return float(sub[col].mean())


def build() -> pd.DataFrame:
    core = pd.read_csv(os.path.join(RES, "core_results.csv"))
    rows = []

    def add(**kw):
        # ensure required keys
        for k in REQUIRED:
            kw.setdefault(k, None)
        rows.append(kw)

    # Cora / PubMed from core
    for defense in ["none", "sami", "gtd", "lbp", "maskarmor"]:
        if defense == "maskarmor" and _m(core, "Cora", "GraphSAGE", defense, "test_accuracy") != _m(
            core, "Cora", "GraphSAGE", defense, "test_accuracy"
        ):
            pass
        acc = _m(core, "Cora", "GraphSAGE", defense, "test_accuracy")
        if acc != acc:
            # maskarmor may be in baselines_extra
            continue
        # Prefer framing JSON for flagship sami
        if defense == "sami":
            fr = json.load(open(os.path.join(RES, "sami_gtd_framing.json")))
            sm = fr["sami_means"]
            add(
                dataset="Cora",
                model="GraphSAGE",
                defense="sami",
                homophily_h=0.81,
                density_rho=0.00144,
                n_nodes=2708,
                split_protocol="random_40_20_40",
                acc=round(sm["test_accuracy"], 4),
                lira_auroc=round(sm["lira_attack_auc"], 4),
                conf_auroc=round(sm["conf_attack_auc"], 4),
                tpr_at_1pct_fpr=round(sm.get("conf_tpr_at_0.01_fpr", float("nan")), 4),
                train_seconds=round(_m(core, "Cora", "GraphSAGE", "sami", "train_seconds"), 2),
                n_shadows=4,
                leakage_band="moderate_reduced",
                notes="locked SAMI; LiRA primary flagship",
            )
            continue
        add(
            dataset="Cora",
            model="GraphSAGE",
            defense=defense,
            homophily_h=0.81,
            density_rho=0.00144,
            n_nodes=2708,
            split_protocol="random_40_20_40",
            acc=round(acc, 4),
            lira_auroc=round(_m(core, "Cora", "GraphSAGE", defense, "lira_attack_auc"), 4),
            conf_auroc=round(_m(core, "Cora", "GraphSAGE", defense, "conf_attack_auc"), 4),
            tpr_at_1pct_fpr=round(
                _m(core, "Cora", "GraphSAGE", defense, "lira_tpr_at_0.01_fpr"), 4
            ),
            train_seconds=round(_m(core, "Cora", "GraphSAGE", defense, "train_seconds"), 2),
            n_shadows=4,
            leakage_band="moderate" if defense == "none" else "defended",
            notes="",
        )

    # MaskArmor: prefer 5-seed confirmatory file (matches paper Table), else baselines_extra
    ma5 = os.path.join(RES, "maskarmor_5seed.csv")
    be = os.path.join(RES, "baselines_extra.csv")
    if not any(r["defense"] == "maskarmor" and r["dataset"] == "Cora" for r in rows):
        sub = None
        notes = "inflates LiRA vs SAMI"
        if os.path.isfile(ma5):
            mdf = pd.read_csv(ma5)
            sub = mdf[(mdf.dataset == "Cora") & (mdf.model == "GraphSAGE") & (mdf.defense == "maskarmor")]
            notes = "5-seed confirmatory (maskarmor_5seed.csv); matches paper Table"
        if (sub is None or len(sub) == 0) and os.path.isfile(be):
            bdf = pd.read_csv(be)
            sub = bdf[(bdf.dataset == "Cora") & (bdf.model == "GraphSAGE") & (bdf.defense == "maskarmor")]
            notes = "baselines_extra; prefer maskarmor_5seed when available"
        if sub is not None and len(sub):
            add(
                dataset="Cora",
                model="GraphSAGE",
                defense="maskarmor",
                homophily_h=0.81,
                density_rho=0.00144,
                n_nodes=2708,
                split_protocol="random_40_20_40",
                acc=round(float(sub.test_accuracy.mean()), 4),
                lira_auroc=round(float(sub.lira_attack_auc.mean()), 4),
                conf_auroc=round(float(sub.conf_attack_auc.mean()), 4),
                tpr_at_1pct_fpr=round(float(sub["lira_tpr_at_0.01_fpr"].mean()), 4),
                train_seconds=round(float(sub.train_seconds.mean()), 2),
                n_shadows=4,
                leakage_band="defended",
                notes=notes,
            )

    add(
        dataset="PubMed",
        model="GraphSAGE",
        defense="none",
        homophily_h=0.80,
        density_rho=0.00028,
        n_nodes=19717,
        split_protocol="random_40_20_40",
        acc=round(_m(core, "PubMed", "GraphSAGE", "none", "test_accuracy"), 4),
        lira_auroc=round(_m(core, "PubMed", "GraphSAGE", "none", "lira_attack_auc"), 4),
        conf_auroc=round(_m(core, "PubMed", "GraphSAGE", "none", "conf_attack_auc"), 4),
        tpr_at_1pct_fpr=round(_m(core, "PubMed", "GraphSAGE", "none", "lira_tpr_at_0.01_fpr"), 4),
        train_seconds=round(_m(core, "PubMed", "GraphSAGE", "none", "train_seconds"), 2),
        n_shadows=4,
        leakage_band="near_chance",
        notes="negative control",
    )

    # Actor full grid
    ab = os.path.join(RES, "actor_baselines.csv")
    if os.path.isfile(ab):
        adf = pd.read_csv(ab)
        for defense, band in [
            ("none", "high"),
            ("gtd", "high"),
            ("lbp", "high"),
            ("maskarmor", "high"),
            ("sami", "high_reduced"),
        ]:
            sub = adf[adf.defense == defense]
            if len(sub) == 0:
                continue
            add(
                dataset="Actor",
                model="GraphSAGE",
                defense=defense,
                homophily_h=0.219,
                density_rho=0.00052,
                n_nodes=7600,
                split_protocol="random_40_20_40",
                acc=round(float(sub.test_accuracy.mean()), 4),
                lira_auroc=round(float(sub.lira_attack_auc.mean()), 4),
                conf_auroc=round(float(sub.conf_attack_auc.mean()), 4),
                tpr_at_1pct_fpr=round(float(sub["lira_tpr_at_0.01_fpr"].mean()), 4),
                train_seconds=round(float(sub.train_seconds.mean()), 2),
                n_shadows=4,
                leakage_band=band,
                notes="heterophilic; Acc low for all methods",
            )

    # Chameleon second heterophilic grid
    ch = os.path.join(RES, "chameleon_baselines.csv")
    if os.path.isfile(ch):
        cdf = pd.read_csv(ch)
        for defense, band in [
            ("none", "high"),
            ("gtd", "moderate"),
            ("lbp", "moderate"),
            ("maskarmor", "high"),
            ("sami", "moderate_reduced"),
        ]:
            sub = cdf[cdf.defense == defense]
            if len(sub) == 0:
                continue
            add(
                dataset="Chameleon",
                model="GraphSAGE",
                defense=defense,
                homophily_h=round(float(cdf.homophily.mean()), 4) if "homophily" in cdf.columns else 0.235,
                density_rho=round(float(cdf.density.mean()), 6) if "density" in cdf.columns else None,
                n_nodes=int(sub.iloc[0].get("n_nodes", 2277)) if "n_nodes" in sub.columns else 2277,
                split_protocol="random_40_20_40",
                acc=round(float(sub.test_accuracy.mean()), 4),
                lira_auroc=round(float(sub.lira_attack_auc.mean()), 4),
                conf_auroc=round(float(sub.conf_attack_auc.mean()), 4),
                tpr_at_1pct_fpr=round(float(sub["lira_tpr_at_0.01_fpr"].mean()), 4),
                train_seconds=round(float(sub.train_seconds.mean()), 2),
                n_shadows=4,
                leakage_band=band,
                notes="second heterophilic real grid; locked SAMI",
            )

    # Citeseer val-selected SAMI + none
    cs = os.path.join(RES, "citeseer_retune_confirm.csv")
    if os.path.isfile(cs):
        sdf = pd.read_csv(cs)
        for variant, defense, notes in [
            ("none", "none", "citeseer confirm"),
            ("sami_selected", "sami", "val-Acc selected λ=1 σ=0.25"),
            ("sami_locked_cora", "sami_lock", "Cora locked λ=0.5 σ=0.35"),
            ("gtd", "gtd", ""),
            ("lbp", "lbp", ""),
            ("maskarmor", "maskarmor", "inflates LiRA"),
        ]:
            sub = sdf[sdf.variant == variant]
            if len(sub) == 0:
                continue
            add(
                dataset="Citeseer",
                model="GraphSAGE",
                defense=defense,
                homophily_h=0.74,
                density_rho=0.00089,
                n_nodes=3327,
                split_protocol="random_40_20_40",
                acc=round(float(sub.test_accuracy.mean()), 4),
                lira_auroc=round(float(sub.lira_attack_auc.mean()), 4),
                conf_auroc=round(float(sub.conf_attack_auc.mean()), 4),
                tpr_at_1pct_fpr=round(float(sub["lira_tpr_at_0.01_fpr"].mean()), 4),
                train_seconds=round(float(sub.train_seconds.mean()), 2) if "train_seconds" in sub.columns else None,
                n_shadows=4,
                leakage_band="moderate_reduced" if "sami" in variant else "moderate",
                notes=notes,
            )

    # Amazon Photo from matched-budget scale ladder (undefended)
    sl = os.path.join(RES, "scale_ladder_dense.csv")
    if not os.path.isfile(sl):
        sl = os.path.join(RES, "scale_ladder.csv")
    if os.path.isfile(sl):
        pdf = pd.read_csv(sl)
        sub = pdf[(pdf.dataset == "Photo") & (pdf.n_shadows == 4)]
        if len(sub):
            add(
                dataset="Photo",
                model="GraphSAGE",
                defense="none",
                homophily_h=round(float(sub.homophily.mean()), 4) if "homophily" in sub.columns else 0.827,
                density_rho=round(float(sub.density.mean()), 6) if "density" in sub.columns else 0.00407,
                n_nodes=7650,
                split_protocol="random_40_20_40",
                acc=round(float(sub.acc.mean()), 4),
                lira_auroc=round(float(sub.lira.mean()), 4),
                conf_auroc=float("nan"),
                tpr_at_1pct_fpr=round(float(sub.tpr01.mean()), 4) if "tpr01" in sub.columns else float("nan"),
                train_seconds=round(float(sub.train_s.mean()), 2) if "train_s" in sub.columns else float("nan"),
                n_shadows=4,
                leakage_band="near_chance",
                notes="matched-budget scale ladder; Photo vs Actor structure control",
            )

    # Amazon Computers canonical run_one grid
    cb = os.path.join(RES, "computers_baselines.csv")
    if os.path.isfile(cb):
        cdf = pd.read_csv(cb)
        for defense, band in [
            ("none", "near_chance"),
            ("gtd", "near_chance"),
            ("lbp", "near_chance"),
            ("maskarmor", "near_chance"),
            ("sami", "near_chance"),
        ]:
            sub = cdf[(cdf.model == "GraphSAGE") & (cdf.defense.astype(str).str.lower() == defense)]
            if len(sub) == 0:
                continue
            add(
                dataset="Computers",
                model="GraphSAGE",
                defense=defense,
                homophily_h=round(float(sub.homophily.mean()), 4) if "homophily" in sub.columns else 0.777,
                density_rho=round(float(sub.density.mean()), 6) if "density" in sub.columns else 0.0026,
                n_nodes=13752,
                split_protocol="random_40_20_40",
                acc=round(float(sub.test_accuracy.mean()), 4),
                lira_auroc=round(float(sub.lira_attack_auc.mean()), 4),
                conf_auroc=round(float(sub.conf_attack_auc.mean()), 4),
                tpr_at_1pct_fpr=round(float(sub["lira_tpr_at_0.01_fpr"].mean()), 4),
                train_seconds=round(float(sub.train_seconds.mean()), 2),
                n_shadows=4,
                leakage_band=band,
                notes="canonical run_one Acc (matches tab:computers / tab:scale)",
            )

    # ogbn
    ogbn = os.path.join(RES, "ogbn_volume_results.csv")
    if os.path.isfile(ogbn):
        odf = pd.read_csv(ogbn)
        for defense in ["none", "sami", "gtd", "lbp"]:
            sub = odf[odf.defense == defense]
            if len(sub) == 0:
                continue
            add(
                dataset="ogbn-arxiv",
                model="GraphSAGE",
                defense=defense,
                homophily_h=round(float(sub.homophily.mean()), 4),
                density_rho=round(float(sub.density.mean()), 6),
                n_nodes=169343,
                split_protocol="ogb_official",
                acc=round(float(sub.test_accuracy.mean()), 4),
                lira_auroc=round(float(sub.lira_attack_auc.mean()), 4),
                conf_auroc=round(float(sub.conf_attack_auc.mean()), 4),
                tpr_at_1pct_fpr=round(float(sub["lira_tpr_at_0.01_fpr"].mean()), 4),
                train_seconds=round(float(sub.train_seconds.mean()), 1),
                n_shadows=2,
                leakage_band="near_chance",
                notes="Volume negative control + systems; api_qps≈27k in ogbn_systems.json",
                api_qps=27236.0 if defense == "none" else None,
            )

    # Stress cell — tagged, demoted
    hrs = os.path.join(RES, "volume_highrisk_synth.csv")
    if os.path.isfile(hrs):
        hdf = pd.read_csv(hrs)
        for defense in hdf.defense.unique():
            sub = hdf[hdf.defense == defense]
            add(
                dataset="synth_stress_n3k",
                model="GCN",
                defense=defense,
                homophily_h=round(float(sub.homophily.mean()), 4),
                density_rho=round(float(sub.density.mean()), 6),
                n_nodes=3000,
                split_protocol="random_40_20_40_stress",
                acc=round(float(sub.test_accuracy.mean()), 4),
                lira_auroc=round(float(sub.lira_attack_auc.mean()), 4),
                conf_auroc=round(float(sub.conf_attack_auc.mean()), 4),
                tpr_at_1pct_fpr=None,
                train_seconds=round(float(sub.train_seconds.mean()), 2)
                if "train_seconds" in sub
                else None,
                n_shadows=4,
                leakage_band="stress_conf_only",
                notes="NOT Volume peer; LiRA≈chance; Acc≈0.29; conf secondary",
            )

    df = pd.DataFrame(rows)
    # Drop rows missing required LiRA/acc
    df = df[df["acc"].notna() & df["lira_auroc"].notna()].copy()
    return df


def plot_regime(df: pd.DataFrame):
    plot = df[df.defense == "none"].copy()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    markers = {
        "ogb_official": "s",
        "random_40_20_40": "o",
        "random_40_20_40_stress": "^",
    }
    for _, r in plot.iterrows():
        x = np.log10(max(r["n_nodes"], 10))
        y = r["lira_auroc"]
        mk = markers.get(r["split_protocol"], "o")
        face = "white" if "stress" in str(r["split_protocol"]) else "#4C78A8"
        ax.scatter(x, y, s=110, marker=mk, facecolors=face, edgecolors="k", linewidths=0.7, zorder=3)
        ax.annotate(
            r["dataset"][:14],
            (x, y),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
        )
    ax.axhline(0.5, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel(r"log$_{10}$(#nodes) [axis for display only]")
    ax.set_ylabel("Undefended LiRA AUROC (primary)")
    ax.set_title("SPAB regimes (split-tagged; stress demoted)")
    ax.set_ylim(0.45, 0.75)
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4C78A8", markeredgecolor="k", markersize=8, label="OGB official"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C78A8", markeredgecolor="k", markersize=8, label="40/20/40"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="white", markeredgecolor="k", markersize=8, label="stress (demoted)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    for d in (FIG, PAPER_VIS):
        fig.savefig(os.path.join(d, "fig_regime_map.png"), dpi=200)
    plt.close(fig)


def main():
    df = build()
    out_csv = os.path.join(RES, "spab_report.csv")
    df.to_csv(out_csv, index=False)
    df.to_json(os.path.join(RES, "spab_report.json"), orient="records", indent=2)
    schema = {
        "name": "SPAB audit report",
        "required_columns": REQUIRED,
        "optional_columns": ["api_qps", "peak_mb", "notes"],
        "note": "Not a blind community benchmark; release artifact matching paper tables. Dropped peak_mb as required.",
        "primary_privacy_metric": "lira_auroc",
    }
    with open(os.path.join(RES, "spab_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)
    snap = os.path.join(RES, "paper_release")
    os.makedirs(snap, exist_ok=True)
    df.to_csv(os.path.join(snap, "spab_report.csv"), index=False)
    df.to_json(os.path.join(snap, "spab_report.json"), orient="records", indent=2)
    with open(os.path.join(snap, "spab_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)
    plot_regime(df)
    # copy LTE ablation if present
    print(f"SPAB rows={len(df)}")
    print(df[["dataset", "defense", "acc", "lira_auroc", "train_seconds", "split_protocol"]].to_string(index=False))


if __name__ == "__main__":
    main()
