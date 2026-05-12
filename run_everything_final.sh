#!/usr/bin/env bash
# Run full experiment grid, then generate all figures.
set -e
cd "$(dirname "$0")"
echo "=== 1. Running full experiment grid (run_final.py) ==="
python3 run_final.py
echo ""
echo "=== 2. Generating figures (generate_figures.py) ==="
python3 generate_figures.py
echo ""
echo "=== Done. Results: results/*.csv; Figures: figures/*.png, figures/*.pdf ==="
