# Deposit this archive on Zenodo

1. Go to https://zenodo.org/deposit/new (login with GitHub/ORCID).
2. Upload: `spab_v1_20260726.zip`
3. Paste fields from `zenodo_staging/zenodo_metadata.json` (title, description, creators, keywords).
4. Click **Publish** → copy DOI `10.5281/zenodo.XXXXXXX`.
5. Optional **Reserve DOI** before publish if you need the number for the camera-ready PDF first.
6. Then run:
   ```bash
   ./venv/bin/python scripts/set_zenodo_doi.py 10.5281/zenodo.XXXXXXX
   ```
   (or edit `ZENODO_README.md` + paper Code Availability).

API alternative (needs token):
```bash
export ZENODO_TOKEN=...
./venv/bin/python scripts/zenodo_deposit.py
```
