# IEEE BigData 2026 — HARP Submission Packet

**Conference:** 2026 IEEE International Conference on Big Data (Phoenix, AZ, Dec 14–17, 2026)  
**Track:** Regular research paper  
**Deadline:** Aug 21, 2026 (AoE)  
**Review policy:** Single-blind (author names required)  
**Freeze commit:** `main` (see `git log -1`)

## Venue checklist

| Requirement | Status |
|---|---|
| IEEE 2-column Computer Society format (`IEEEtran` conference) | ✅ |
| ≤ 10 pages **including references** | ✅ **10 pages** |
| Single-blind (names + affiliations on title page) | ✅ |
| No Overfull `\hbox` | ✅ |
| Figures as vector PDF (sharp in PDF viewers) | ✅ |
| Original unpublished work | ✅ (author responsibility) |
| Submit via CyberChair | [Submission portal](https://wi-lab.com/cyberchair/2026/bigdata26/index.php) |
| Formatting guidelines | [IEEE templates](https://www.ieee.org/conferences/publishing/templates.html) |

## Upload this file

```
submission/HARP_IEEE_BigData_2026.pdf
```

(identical to `paper/ieee_privacy_gnn.pdf`)

## Title / authors (as on PDF)

**Title:** HARP: ExactFrac-Constrained Score Serving for GNN Prediction APIs  

**Authors:**  
1. Krish Sharma — AMSAT / Ed W. Clark High School & DiSC, UNLV  
2. Shaikh Arifuzzaman — DiSC, Dept. of Computer Science, UNLV  

## Suggested CyberChair topics

- Big Data Infrastructure / Software Systems to Support Big Data Computing  
- Privacy preserving Big Data collection/analytics  
- Trust, resilience, privacy, and security issues  
- Data and Information Quality for Big Data  
- Cloud/Grid/Stream Computing for Big Data  

## Artifact freeze

| Item | Path |
|---|---|
| Manuscript TeX | `paper/ieee_privacy_gnn.tex` |
| Manuscript PDF | `paper/ieee_privacy_gnn.pdf` |
| Figures | `fig_harp_schematic`, `fig_harp_exactfrac_pareto`, `fig_harp_headline`, `fig_harp_nsh16_scatter` |
| Reproduce guide | `REPRODUCE.md` |
| Locked config | `LOCKED_HARP_RELEASE` in `defenses/harp.py` |
| Headline CSVs | `results/harp_headline_nsh16*.csv`, `harp_cfs_nsh16*.csv`, `harp_exactfrac_sla_evidence*.csv` |
| Code | https://github.com/krishharma/privacy-gnn |

## Pre-submit verification

```bash
cd paper && pdflatex ieee_privacy_gnn.tex && pdflatex ieee_privacy_gnn.tex
# Confirm: Output written ... (10 pages, ...)
./venv/bin/python -m pytest tests/test_harp.py -q
```

## Upload notes

1. Select **Regular Paper**.  
2. Upload the **10-page PDF** only (GitHub URL is in the paper).  
3. Do **not** anonymize — single-blind.  
4. Camera-ready (if accepted) due Nov 14, 2026.  
