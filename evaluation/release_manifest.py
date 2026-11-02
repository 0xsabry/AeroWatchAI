"""AeroWatch AI — release manifest generator (Phase 10).

    python evaluation/release_manifest.py --test-summary "163 passed, 1 skipped"

Writes docs/RELEASE_MANIFEST.json with machine-derived facts only:
git commit, Python version, lockfile + checkpoint + ONNX SHA-256,
default backend, platform. The test summary is supplied explicitly
from the verification run that produced the release — never guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True,
                              timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build(test_summary: str) -> dict:
    return {
        "project": "AeroWatchAI",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "requirements_lock_sha256": _sha256(
            PROJECT_ROOT / "requirements-lock.txt"),
        "checkpoint_v7_sha256": _sha256(
            PROJECT_ROOT / "weights" / "aerowatch_v7_balanced.pt"),
        "onnx_artifact_sha256": _sha256(
            PROJECT_ROOT / "weights" / "aerowatch_v7_balanced.onnx"),
        "default_backend": "pytorch",
        "optional_backend": "onnx (opt-in)",
        "default_checkpoint": "weights/aerowatch_v7_balanced.pt",
        "test_summary": test_summary,
        "release_status": {
            "technical_verification": "TECHNICALLY VERIFIED",
            "owner_validation": "OWNER VALIDATION PENDING",
            "distribution": "DISTRIBUTION NOT CLEARED",
            "hardware_verification": (
                "NOT VERIFIED — RTX 3050 benchmark is an owner-side "
                "run (evaluation/hardware_benchmark.py)"),
            "camera_verification": (
                "NOT VERIFIED — no physical camera in the development "
                "environment (evaluation/camera_check.py --model)"),
            "licensing": (
                "UNRESOLVED — see docs/LICENSING_DECISION.md; LICENSE "
                "unchanged; no commercial clearance claimed"),
            "ci_workflow": (
                "NOT APPLIED — token lacks workflow scope; owner "
                "copies docs/ci_workflow_phase4.yml to "
                ".github/workflows/ci.yml"),
        },
        "known_restrictions": [
            "Licensing unresolved (docs/LICENSING_DECISION.md): "
            "not cleared for commercial distribution; weights are "
            "not redistributed by this project",
            "No authentication: localhost or authenticating proxy only",
            "RTX 3050 GPU figures not yet measured (owner-side run)",
            "Physical camera verification pending (owner-side run)",
            "GitHub Actions workflow not applied (token lacks "
            "workflow scope; docs/ci_workflow_phase4.yml preserved)",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-summary", required=True)
    args = parser.parse_args()
    manifest = build(args.test_summary)
    out = PROJECT_ROOT / "docs" / "RELEASE_MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest written: {out}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
