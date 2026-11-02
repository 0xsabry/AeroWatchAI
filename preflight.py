"""AeroWatch AI — environment preflight checks (Phase 10).

Cross-platform checks used by the Windows setup/launcher scripts and
runnable standalone anywhere:

    python preflight.py [--json]

Verifies: Python version, core dependencies importable, the default
checkpoint's presence and SHA-256, the ONNX artifact (optional),
CPU/GPU environment, and writable history storage. Every check
reports ok/warn/fail with a human-actionable message. Nothing is
downloaded and nothing is modified.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_V7_SHA256 = (
    "992080afb8f38674ddbf4eb84b9f8123b89bb4f90965c52e6c5e58af328b51cb"
)
MIN_PYTHON = (3, 10)
CORE_MODULES = ["streamlit", "cv2", "numpy", "pandas", "PIL", "torch",
                "ultralytics"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_preflight(root: Path = PROJECT_ROOT) -> Dict:
    checks: List[Dict] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    py = sys.version_info
    add("python_version",
        "ok" if py[:2] >= MIN_PYTHON else "fail",
        f"Python {platform.python_version()} "
        f"(need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}; 3.12 verified)")

    missing = []
    for mod in CORE_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            missing.append(f"{mod} ({exc.__class__.__name__})")
    add("dependencies", "fail" if missing else "ok",
        "missing: " + ", ".join(missing) if missing
        else "all core modules importable")

    ckpt = root / "weights" / "aerowatch_v7_balanced.pt"
    if not ckpt.exists():
        add("checkpoint_v7", "fail",
            f"not found: {ckpt} — obtain the project weights; the app "
            "cannot detect without them and they are not downloaded "
            "automatically")
    else:
        digest = _sha256(ckpt)
        add("checkpoint_v7",
            "ok" if digest == EXPECTED_V7_SHA256 else "fail",
            f"sha256 {digest[:16]}… "
            + ("matches the verified release hash"
               if digest == EXPECTED_V7_SHA256
               else "DOES NOT match the verified release hash — "
                    "do not use this file"))

    onnx = root / "weights" / "aerowatch_v7_balanced.onnx"
    add("onnx_artifact", "ok" if onnx.exists() else "warn",
        "present (optional ONNX backend available)" if onnx.exists()
        else "absent — optional; the default PyTorch backend is "
             "unaffected")

    gpu_detail = "CPU only"
    gpu_status = "warn"
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_detail = (f"{props.name}, {props.total_memory / 1e9:.1f} GB "
                          f"VRAM, torch CUDA {torch.version.cuda}")
            gpu_status = "ok"
        else:
            gpu_detail = ("no CUDA GPU detected — CPU inference works "
                          "(slower); see docs/WINDOWS_GUIDE.md for the "
                          "GPU install path")
    except Exception as exc:  # noqa: BLE001
        gpu_detail = f"torch GPU probe failed: {exc}"
    add("compute_device", gpu_status, gpu_detail)

    hist_dir = Path(os.environ.get(
        "AEROWATCH_HISTORY_DIR", root / "runs" / "history"))
    try:
        hist_dir.mkdir(parents=True, exist_ok=True)
        probe = hist_dir / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        add("history_storage", "ok", f"writable: {hist_dir}")
    except OSError as exc:
        add("history_storage", "fail", f"not writable: {hist_dir} ({exc})")

    overall = ("fail" if any(c["status"] == "fail" for c in checks)
               else "ok")
    return {"overall": overall, "checks": checks}


def main() -> int:
    report = run_preflight()
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2))
    else:
        for c in report["checks"]:
            icon = {"ok": "PASS", "warn": "WARN", "fail": "FAIL"}[c["status"]]
            print(f"[{icon}] {c['name']}: {c['detail']}")
        print(f"\nOverall: {report['overall'].upper()}")
    return 0 if report["overall"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
