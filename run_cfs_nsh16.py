#!/usr/bin/env python3
"""CFS under headline LiRA budget n_shadows=16 (fixes n_sh=4 / tau=0.55 inconsistency)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from config import load_config
from defenses.harp import LOCKED_HARP_RELEASE, constrained_frac_search
from experiment import run_one

OUT = "results"
SEEDS = [42, 123, 456]  # 3 seeds × grid is the modern-budget CFS study
FRAC_GRID = [0.0, 0.20, 0.30, 0.40]
# Modern LiRA: locked HARP≈0.67, none≈0.81. tau=0.70 is a deployable audit cap.
TAU = 0.70
C = 0.60


def main():
    cfg = load_config("experiment_config_confirmatory.yaml")
    cfg["lira"] = {"n_shadows": 16}
    grid_path = os.path.join(OUT, "harp_cfs_nsh16_grid.csv")
    best_path = os.path.join(OUT, "harp_cfs_nsh16.csv")
    grid_rows = pd.read_csv(grid_path).to_dict("records") if os.path.isfile(grid_path) else []
    best_rows = pd.read_csv(best_path).to_dict("records") if os.path.isfile(best_path) else []
    done_g = {(r["constructor"], int(r["seed"]), round(float(r["Frac"]), 3)) for r in grid_rows}
    done_b = {(r["constructor"], int(r["seed"]), float(r["tau"])) for r in best_rows}

    for mode, dn, extra in [
        ("topology", "harp_release_only", {}),
        ("ensemble", "harp_ensemble", {"seed_mode": "ensemble"}),
        ("random", "harp_random", {"seed_mode": "random"}),
    ]:
        for seed in SEEDS:
            cache = {}
            for frac in FRAC_GRID:
                key = (mode, seed, round(float(frac), 3))
                if key in done_g:
                    hit = [r for r in grid_rows if r["constructor"] == mode and int(r["seed"]) == seed and abs(float(r["Frac"]) - frac) < 1e-9][0]
                    cache[round(float(frac), 3)] = {
                        "Acc": float(hit["Acc"]),
                        "LiRA": float(hit["LiRA"]),
                        "ExactFrac": float(hit["ExactFrac"]),
                        "Mass": hit.get("Mass"),
                        "TPR1": hit.get("TPR1"),
                    }
                    continue
                print(f"CFS16 {mode} seed={seed} frac={frac}", flush=True)
                params = {**LOCKED_HARP_RELEASE, "target_protect_frac": float(frac), **extra}
                if frac <= 0:
                    # ExactFrac=1 point: no protector
                    r = run_one("Cora", "GraphSAGE", "none", {}, seed, config=cfg)
                    out = {
                        "Acc": float(r["test_accuracy"]),
                        "LiRA": float(r["lira_attack_auc"]),
                        "ExactFrac": 1.0,
                        "Mass": 0.0,
                        "TPR1": float(r.get("lira_tpr_at_0.01_fpr", np.nan)),
                    }
                else:
                    r = run_one("Cora", "GraphSAGE", dn, params, seed, config=cfg)
                    out = {
                        "Acc": float(r["test_accuracy"]),
                        "LiRA": float(r["lira_attack_auc"]),
                        "ExactFrac": float(r.get("exact_frac", 1.0 - frac)),
                        "Mass": r.get("noise_mass"),
                        "TPR1": float(r.get("lira_tpr_at_0.01_fpr", np.nan)),
                    }
                cache[round(float(frac), 3)] = out
                grid_rows.append({
                    "constructor": mode, "seed": seed, "Frac": frac,
                    "n_shadows": 16, "c": C, **out,
                })
                done_g.add(key)
                pd.DataFrame(grid_rows).to_csv(grid_path, index=False)
                print(f"  Acc={out['Acc']:.3f} LiRA={out['LiRA']:.3f} EF={out['ExactFrac']:.3f}", flush=True)

            def eval_frac(frac, cache=cache):
                return dict(cache[round(float(frac), 3)])

            for tau in (TAU, 0.68, 0.75):
                if (mode, seed, tau) in done_b:
                    continue
                best = constrained_frac_search(
                    eval_frac, exact_frac_min=C, lira_max=tau, frac_grid=FRAC_GRID,
                )
                best.update({
                    "dataset": "Cora", "seed": seed, "constructor": mode,
                    "n_shadows": 16, "tau": tau,
                })
                best_rows.append(best)
                done_b.add((mode, seed, tau))
                pd.DataFrame(best_rows).to_csv(best_path, index=False)
                print(f"BEST tau={tau} {mode} seed={seed}: Frac={best.get('Frac')} Acc={best.get('Acc')} LiRA={best.get('LiRA')} feas={best.get('feasible_audit')}", flush=True)

    gdf = pd.DataFrame(grid_rows)
    print("\n=== GRID MEANS ===", flush=True)
    print(gdf.groupby(["constructor", "Frac"])[["Acc", "LiRA", "ExactFrac"]].mean().round(4), flush=True)
    bdf = pd.DataFrame(best_rows)
    print("\n=== CFS BEST MEANS ===", flush=True)
    print(bdf.groupby(["tau", "constructor"])[["Frac", "Acc", "LiRA", "feasible_audit"]].mean().round(4), flush=True)
    print("CFS16 DONE", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PRIVACYGNN_CONFIG", "experiment_config_confirmatory.yaml")
    main()
