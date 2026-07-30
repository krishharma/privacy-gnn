"""Fill Table tab:harp in ieee_privacy_gnn.tex from results/harp_means.csv."""
from __future__ import annotations

import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEANS = os.path.join(ROOT, "results", "harp_means.csv")
TEX = os.path.join(ROOT, "paper", "ieee_privacy_gnn.tex")

ORDER = [
    ("Cora", "none"),
    ("Cora", "lbp"),
    ("Cora", "sami"),
    ("Cora", "harp"),
    ("Cora", "harp_k0"),
    ("Cora", "harp_release_only"),
    ("Citeseer", "none"),
    ("Citeseer", "lbp"),
    ("Citeseer", "harp"),
    ("Chameleon", "none"),
    ("Chameleon", "lbp"),
    ("Chameleon", "harp"),
    ("Actor", "none"),
    ("Actor", "lbp"),
    ("Actor", "harp"),
]

LABEL = {
    "none": "none",
    "lbp": "LBP",
    "sami": "SAMI",
    "harp": "HARP",
    "harp_k0": "HARP $k{=}0$",
    "harp_k2": "HARP $k{=}2$",
    "harp_release_only": "HARP release-only",
    "gtd": "GTD",
}


def fmt(x, nd=3):
    if x != x:
        return "---"
    return f"{float(x):.{nd}f}"


def main():
    if not os.path.isfile(MEANS):
        raise SystemExit(f"missing {MEANS}")
    m = pd.read_csv(MEANS)
    m = m[m.model == "GraphSAGE"]
    rows = []
    for ds, dn in ORDER:
        sub = m[(m.dataset == ds) & (m.defense == dn)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        mass = r.get("noise_mass", float("nan"))
        frac = r.get("frac_protected", float("nan"))
        rows.append(
            f"{ds} & {LABEL.get(dn, dn)} & {fmt(r.test_accuracy)} & "
            f"{fmt(r.lira_attack_auc)} & {fmt(mass, 0)} & {fmt(frac, 2)} \\\\"
        )
    body = "\n".join(rows)
    table = f"""\\begin{{table}}[!t]
\\caption{{HARP primary grid (GraphSAGE; means over seeds). Mass$=\\sum\\sigma_v$; Frac$=$protected fraction.}}
\\label{{tab:harp}}
\\centering
\\scriptsize
\\begin{{tabular}}{{@{{}}llcccc@{{}}}}
\\toprule
\\textbf{{Dataset}} & \\textbf{{Defense}} & \\textbf{{Acc}}$\\uparrow$ & \\textbf{{LiRA}}$\\downarrow$ & \\textbf{{Mass}}$\\downarrow$ & \\textbf{{Frac}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    tex = open(TEX).read()
    pat = re.compile(
        r"\\begin\{table\}\[!t\]\n\\caption\{HARP primary grid.*?\\end\{table\}",
        re.S,
    )
    if not pat.search(tex):
        raise SystemExit("tab:harp not found")
    open(TEX, "w").write(pat.sub(lambda _: table, tex, count=1))
    print("Updated tab:harp with", len(rows), "rows")


if __name__ == "__main__":
    main()
