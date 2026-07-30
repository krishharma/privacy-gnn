"""Write Zenodo DOI into ZENODO_README.md and paper/ieee_privacy_gnn.tex."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 2:
        print("Usage: set_zenodo_doi.py 10.5281/zenodo.XXXXXXX")
        sys.exit(1)
    doi = sys.argv[1].strip()
    if not doi.startswith("10.5281/zenodo."):
        print("Expected DOI like 10.5281/zenodo.1234567")
        sys.exit(1)
    url = f"https://doi.org/{doi}"

    readme = ROOT / "ZENODO_README.md"
    if readme.is_file():
        t = readme.read_text()
        t = re.sub(
            r"https://doi\.org/10\.5281/zenodo\.\w+|DOI placeholder:.*",
            url,
            t,
        )
        if url not in t:
            t = t.replace(
                "Replace this line with the Zenodo DOI after deposit:\n`https://doi.org/10.5281/zenodo.XXXXXXXX`",
                f"Zenodo DOI:\n`{url}`",
            )
        readme.write_text(t)
        print("updated", readme)

    tex = ROOT / "paper" / "ieee_privacy_gnn.tex"
    if tex.is_file():
        t = tex.read_text()
        # Code availability / artifact lines
        if "doi.org/10.5281/zenodo" in t:
            t = re.sub(r"https://doi\.org/10\.5281/zenodo\.\w+", url, t)
        elif "Zenodo deposit accompanies" in t:
            t = t.replace(
                "Zenodo deposit accompanies the camera-ready.",
                f"Artifact DOI: \\url{{{url}}}.",
            )
        else:
            t = t.replace(
                r"See \texttt{REPRODUCE.md} and \texttt{EVALUATION\_PROTOCOL.md}.",
                r"See \texttt{REPRODUCE.md} and \texttt{EVALUATION\_PROTOCOL.md}. "
                f"Artifact: \\url{{{url}}}.",
            )
        tex.write_text(t)
        print("updated", tex)

    meta = ROOT / "results" / "zenodo_doi.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text('{\n  "doi": "%s",\n  "url": "%s"\n}\n' % (doi, url))
    print("wrote", meta)


if __name__ == "__main__":
    main()
