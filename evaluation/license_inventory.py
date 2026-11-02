"""AeroWatch AI — automated software license inventory.

Generates an inventory of installed Python distributions from package
metadata (importlib.metadata): name, version, declared license string,
license classifiers, and whether a LICENSE/COPYING file ships inside
the distribution.

This is *evidence collection*, not a legal assessment: a non-empty
"license" field means the package declares something; it does not by
itself establish what your obligations are. The analysis lives in
docs/LICENSE_AUDIT.md.

Usage:
    python evaluation/license_inventory.py \
        [--out docs/licenses_inventory.json] [--filter requirements]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def dist_record(dist: metadata.Distribution) -> dict:
    md = dist.metadata
    classifiers = [
        c for c in (md.get_all("Classifier") or []) if "License" in c
    ]
    license_files = []
    if dist.files:
        for f in dist.files:
            name = Path(str(f)).name.upper()
            if name.startswith(("LICENSE", "COPYING", "NOTICE")) and len(license_files) < 5:
                license_files.append(str(f))
    return {
        "name": md.get("Name") or dist.name,
        "version": dist.version,
        "license_declared": (md.get("License") or "").strip()[:300],
        "license_classifiers": classifiers,
        "license_files": license_files,
        "home_page": md.get("Home-page") or md.get("Project-URL") or "",
    }


def requirements_names() -> set[str]:
    req = PROJECT_ROOT / "requirements.txt"
    names = set()
    if req.exists():
        for line in req.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.add(line.split("=")[0].split(">")[0].split("<")[0].strip().lower())
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "licenses_inventory.json"))
    ap.add_argument("--scope", choices=["all", "requirements"], default="requirements",
                    help="requirements = distributions matching requirements.txt (+ their key runtime deps)")
    args = ap.parse_args()

    dists = [dist_record(d) for d in metadata.distributions()]
    dists.sort(key=lambda r: r["name"].lower())
    if args.scope == "requirements":
        wanted = requirements_names() | {"torch", "torchvision", "ultralytics", "opencv-python",
                                         "streamlit", "numpy", "pillow", "pandas", "pyyaml"}
        dists = [d for d in dists if d["name"].lower().replace("_", "-") in wanted]
    payload = {
        "generated": date.today().isoformat(),
        "generator": "evaluation/license_inventory.py (importlib.metadata of the active environment)",
        "scope": args.scope,
        "count": len(dists),
        "distributions": dists,
        "note": "Declared metadata only — see docs/LICENSE_AUDIT.md for analysis. Absence of a declared license is itself a finding.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    for d in dists:
        lic = d["license_declared"] or "(none declared)"
        print(f"{d['name']:28s} {d['version']:14s} {lic[:60]}")
    missing = [d["name"] for d in dists if not d["license_declared"] and not d["license_classifiers"]]
    print(f"\n{len(dists)} distributions; no license metadata at all: {missing}")
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
