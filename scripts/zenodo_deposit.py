"""
Optional Zenodo deposit via REST API.
Requires: export ZENODO_TOKEN=... (create at https://zenodo.org/account/settings/applications/)
Uses sandbox if ZENODO_SANDBOX=1.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "zenodo_staging"
OUT = ROOT / "results" / "spab_v1_zenodo"


def main():
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("Set ZENODO_TOKEN. Archive is ready under results/spab_v1_zenodo/ — upload via web UI.")
        sys.exit(2)
    sandbox = os.environ.get("ZENODO_SANDBOX", "").lower() in ("1", "true", "yes")
    base = "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"

    zips = sorted(OUT.glob("spab_v1_*.zip"))
    if not zips:
        print("No zip; run bash build_zenodo_package.sh first")
        sys.exit(1)
    zip_path = zips[-1]
    meta_path = STAGING / "zenodo_metadata.json"
    if not meta_path.is_file():
        print("missing", meta_path)
        sys.exit(1)
    meta = json.loads(meta_path.read_text())

    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(f"{base}/deposit/depositions", json={}, headers=headers, timeout=60)
    r.raise_for_status()
    dep = r.json()
    depo_id = dep["id"]
    bucket = dep["links"]["bucket"]
    print("created deposition", depo_id)

    with open(zip_path, "rb") as f:
        rr = requests.put(
            f"{bucket}/{zip_path.name}",
            data=f,
            headers={**headers, "Content-Type": "application/octet-stream"},
            timeout=600,
        )
    rr.raise_for_status()
    print("uploaded", zip_path.name)

    data = {"metadata": meta}
    r2 = requests.put(
        f"{base}/deposit/depositions/{depo_id}",
        data=json.dumps(data),
        headers={**headers, "Content-Type": "application/json"},
        timeout=60,
    )
    r2.raise_for_status()
    print("metadata set")

    if os.environ.get("ZENODO_PUBLISH", "").lower() in ("1", "true", "yes"):
        r3 = requests.post(
            f"{base}/deposit/depositions/{depo_id}/actions/publish",
            headers=headers,
            timeout=60,
        )
        r3.raise_for_status()
        pub = r3.json()
        doi = pub.get("doi") or pub.get("metadata", {}).get("doi")
        print("PUBLISHED doi=", doi)
        if doi:
            os.system(f'{sys.executable} {ROOT / "scripts" / "set_zenodo_doi.py"} {doi}')
    else:
        # reserve concept DOI if present
        doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi")
        print("Draft ready. Prereserve DOI:", doi)
        print("Publish on Zenodo UI or re-run with ZENODO_PUBLISH=1")
        out = {"deposition_id": depo_id, "prereserve_doi": doi, "sandbox": sandbox}
        (OUT / "zenodo_deposit_draft.json").write_text(json.dumps(out, indent=2))
        if doi:
            os.system(f'{sys.executable} {ROOT / "scripts" / "set_zenodo_doi.py"} {doi}')


if __name__ == "__main__":
    main()
