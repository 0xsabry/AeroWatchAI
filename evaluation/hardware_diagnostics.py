"""AeroWatch AI — one-command hardware diagnostics (Phase 10).

    python evaluation/hardware_diagnostics.py [--json-out PATH]

Reports what is *actually* present on the machine it runs on: CPU,
system RAM, GPU identity + NVIDIA driver (via nvidia-smi when
available), CUDA availability, PyTorch/CUDA compatibility, VRAM,
ONNX Runtime providers, checkpoint integrity, and a recommended
inference configuration. On a machine without a GPU the GPU fields
report "unavailable" — nothing is invented. Pair with
`evaluation/hardware_benchmark.py` for measured inference numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_V7_SHA256 = (
    "992080afb8f38674ddbf4eb84b9f8123b89bb4f90965c52e6c5e58af328b51cb"
)


def _cpu_model() -> str:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round(pages * page_size / 1e9, 1)
    except (ValueError, OSError):
        pass
    return None


def _nvidia_smi() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"present": False}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()
        name, driver, mem = [p.strip() for p in out[0].split(",")]
        return {"present": True, "gpu": name, "driver": driver,
                "vram": mem}
    except Exception as exc:  # noqa: BLE001
        return {"present": True, "error": str(exc)}


def collect() -> dict:
    report: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "nvidia_smi": _nvidia_smi(),
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            report["gpu_name"] = props.name
            report["vram_gb"] = round(props.total_memory / 1e9, 2)
            report["cuda_capability"] = f"{props.major}.{props.minor}"
        else:
            report["gpu_name"] = None
            report["vram_gb"] = None
    except Exception as exc:  # noqa: BLE001
        report["torch_error"] = str(exc)
        report["cuda_available"] = False
    try:
        import onnxruntime as ort

        report["onnxruntime"] = ort.__version__
        report["ort_providers"] = ort.get_available_providers()
    except Exception:  # noqa: BLE001
        report["onnxruntime"] = None
        report["ort_providers"] = []

    ckpt = PROJECT_ROOT / "weights" / "aerowatch_v7_balanced.pt"
    if ckpt.exists():
        digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        report["checkpoint_sha256"] = digest
        report["checkpoint_verified"] = digest == EXPECTED_V7_SHA256
    else:
        report["checkpoint_sha256"] = None
        report["checkpoint_verified"] = False

    cuda = report.get("cuda_available", False)
    report["recommended_config"] = {
        "backend": "pytorch (default)",
        "device": "cuda" if cuda else "cpu",
        "imgsz": 640,
        "precision": "fp16" if cuda else "fp32",
        "note": ("640 is the checkpoint's trained resolution and its "
                 "measured best operating point on every backend "
                 "tested; do not raise it expecting better accuracy."),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    report = collect()
    for key, value in report.items():
        print(f"{key}: {value}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2),
                                       encoding="utf-8")
        print(f"\nJSON written: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
