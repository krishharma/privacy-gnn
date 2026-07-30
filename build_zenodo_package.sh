#!/usr/bin/env bash
# Build a Zenodo-ready archive for SPAB v1.0 / IEEE BigData 2026 artifact.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
STAGING="$ROOT/zenodo_staging"
OUT="$ROOT/results/spab_v1_zenodo"
rm -rf "$STAGING"
mkdir -p "$STAGING" "$OUT"

# Spec + protocol
cp "$ROOT/SPAB_v1_SPEC.md" "$STAGING/"
cp "$ROOT/ZENODO_README.md" "$STAGING/"
cp "$ROOT/EVALUATION_PROTOCOL.md" "$STAGING/" 2>/dev/null || true
cp "$ROOT/REPRODUCE.md" "$STAGING/" 2>/dev/null || true
cp "$ROOT/README.md" "$STAGING/" 2>/dev/null || true
cp "$ROOT/LICENSE" "$STAGING/" 2>/dev/null || true

# Frozen release
mkdir -p "$STAGING/paper_release"
if [[ -d "$ROOT/results/paper_release" ]]; then
  rsync -a --exclude '*.pdf' "$ROOT/results/paper_release/" "$STAGING/paper_release/" || \
    cp -R "$ROOT/results/paper_release/"* "$STAGING/paper_release/" 2>/dev/null || true
fi
cp "$ROOT/results/spab_report.csv" "$STAGING/" 2>/dev/null || true
cp "$ROOT/results/spab_schema.json" "$STAGING/" 2>/dev/null || true
cp "$ROOT/paper/ieee_privacy_gnn.pdf" "$STAGING/ieee_privacy_gnn.pdf" 2>/dev/null || true

# Key runners (source only)
mkdir -p "$STAGING/scripts"
for f in \
  build_spab_release.py freeze_paper_release.py \
  run_ogbn_lira_n4.py run_planetoid_appendix.py run_citeseer_retune.py \
  run_chameleon_baselines.py run_lte_quintile_attribution.py \
  run_actor_highsigma_retune.py run_ogbn_products_volume.py \
  run_systems_audit_cost.py run_actor_baselines.py run_ogbn_lira_n16.py run_arxiv_year_lira.py
do
  [[ -f "$ROOT/$f" ]] && cp "$ROOT/$f" "$STAGING/scripts/"
done

# Metadata for Zenodo web form / API
cat > "$STAGING/zenodo_metadata.json" <<'EOF'
{
  "title": "SPAB v1.0: A Scale-Aware Privacy Audit Protocol for Graph Neural Network Prediction APIs",
  "upload_type": "dataset",
  "description": "Audit protocol, frozen SPAB CSV, and reproduction scripts accompanying the IEEE BigData 2026 paper on scale-aware conditional membership-privacy audits for graph neural network prediction APIs. Primary metric: defense-aware LiRA. Includes ogbn-arxiv systems + negative control, Cora/Citeseer/Chameleon/Actor grids, and locked SAMI configs.",
  "creators": [
    {"name": "Sharma, Krish", "affiliation": "Ed W. Clark High School / UNLV"},
    {"name": "Arifuzzaman, Shaikh", "affiliation": "University of Nevada, Las Vegas"}
  ],
  "keywords": [
    "graph neural networks",
    "membership inference",
    "privacy audit",
    "LiRA",
    "ogbn-arxiv",
    "SPAB"
  ],
  "license": "mit",
  "access_right": "open",
  "communities": [],
  "related_identifiers": []
}
EOF

DATE=$(date +%Y%m%d)
ZIP="$OUT/spab_v1_${DATE}.zip"
( cd "$STAGING" && zip -r -q "$ZIP" . )
echo "Wrote $ZIP"
ls -lh "$ZIP"
# checksum
( cd "$OUT" && shasum -a 256 "$(basename "$ZIP")" > "$(basename "$ZIP").sha256" )
cat "$OUT/$(basename "$ZIP").sha256"

# Deposit helper
cat > "$OUT/DEPOSIT.md" <<EOF
# Deposit this archive on Zenodo

1. Go to https://zenodo.org/deposit/new (login with GitHub/ORCID).
2. Upload: \`$(basename "$ZIP")\`
3. Paste fields from \`zenodo_staging/zenodo_metadata.json\` (title, description, creators, keywords).
4. Click **Publish** → copy DOI \`10.5281/zenodo.XXXXXXX\`.
5. Optional **Reserve DOI** before publish if you need the number for the camera-ready PDF first.
6. Then run:
   \`\`\`bash
   ./venv/bin/python scripts/set_zenodo_doi.py 10.5281/zenodo.XXXXXXX
   \`\`\`
   (or edit \`ZENODO_README.md\` + paper Code Availability).

API alternative (needs token):
\`\`\`bash
export ZENODO_TOKEN=...
./venv/bin/python scripts/zenodo_deposit.py
\`\`\`
EOF
echo "See $OUT/DEPOSIT.md"
