"""AeroWatch AI — unified release validation (Phase 11).

One entry point that orchestrates the existing tools and writes a
machine-readable + human-readable report to ``release_validation/``:

    python release_validation.py                 # quick validation
    python release_validation.py --full          # + hardware benchmark
    python release_validation.py --camera        # + physical camera check
    python release_validation.py --offline       # + offline-mode inference

Statuses are PASS / FAIL / NOT VERIFIED. Hardware or camera checks
that were not requested — or whose hardware is absent — are reported
NOT VERIFIED, never as passes. Expensive benchmarks and camera access
run only in their explicit modes. The report contains no credentials,
no personal identifiers, and no absolute filesystem paths.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS, FAIL, NOT_VERIFIED = "PASS", "FAIL", "NOT VERIFIED"


def _section(name: str, status: str, detail: str, **extra) -> Dict:
    return {"name": name, "status": status, "detail": detail, **extra}


def check_preflight() -> Dict:
    from preflight import run_preflight

    report = run_preflight()
    failed = [c["name"] for c in report["checks"] if c["status"] == "fail"]
    return _section(
        "preflight", PASS if report["overall"] == "ok" else FAIL,
        "all checks pass" if not failed else f"failing: {failed}",
        checks=report["checks"])


def check_dependencies() -> Dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True, text=True, timeout=300)
    ok = proc.returncode == 0
    return _section("dependencies", PASS if ok else FAIL,
                    (proc.stdout or proc.stderr).strip()[:300]
                    or "pip check clean")


def check_hardware() -> Dict:
    from evaluation.hardware_diagnostics import collect

    info = collect()
    return _section(
        "hardware_info", PASS,
        f"cuda_available={info.get('cuda_available')}; "
        f"gpu={info.get('gpu_name') or 'none detected'}",
        hardware={k: info.get(k) for k in (
            "cpu", "cpu_cores", "ram_gb", "gpu_name", "vram_gb",
            "torch", "torch_cuda_build", "onnxruntime", "ort_providers",
            "checkpoint_verified", "recommended_config")})


def check_onnx_smoke() -> Dict:
    onnx_path = PROJECT_ROOT / "weights" / "aerowatch_v7_balanced.onnx"
    if not onnx_path.exists():
        return _section("onnx_smoke", NOT_VERIFIED,
                        "ONNX artifact absent; PyTorch default unaffected")
    try:
        from onnx_backend import OnnxYoloWrapper

        wrapper = OnnxYoloWrapper(str(onnx_path))
        names = wrapper.names
        ok = bool(names) and "drone" in names.values()
        return _section("onnx_smoke", PASS if ok else FAIL,
                        f"loaded; classes={sorted(names.values())}")
    except Exception as exc:  # noqa: BLE001
        return _section("onnx_smoke", FAIL, f"load failed: {exc}")


def check_database() -> Dict:
    from history_store import HistoryStore

    with tempfile.TemporaryDirectory() as tmp:
        store = HistoryStore(Path(tmp) / "validation.db")
        sid = store.start_session(kind="image",
                                  source_label="validation.jpg")
        store.add_detections(sid, [{
            "internal_name": "drone", "display_label": "Drone",
            "conf": 0.5, "posture": "requires_verification",
            "box": [0, 0, 4, 4]}])
        backup = store.backup_to(Path(tmp) / "backup.db")
        store.clear_all()
        store.restore_from(backup, confirm=True)
        info = store.verify_integrity()
        ok = info["ok"] and store.totals()["sessions"] == 1
        store.close()
    return _section("database", PASS if ok else FAIL,
                    "roundtrip: write → backup → clear → restore → "
                    f"integrity {info['integrity']}")


def check_app_startup() -> Dict:
    port = 8791
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true", "--server.port", str(port)],
        cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                        f"http://localhost:{port}/healthz", timeout=3) as r:
                    if r.status == 200:
                        return _section("app_startup", PASS,
                                        "Streamlit health endpoint 200")
            except Exception:  # noqa: BLE001 - still booting
                time.sleep(2)
        return _section("app_startup", FAIL, "health endpoint not reached")
    finally:
        import os
        import signal

        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:  # noqa: BLE001
            proc.terminate()


def check_benchmark(full: bool) -> Dict:
    if not full:
        return _section("hardware_benchmark", NOT_VERIFIED,
                        "not requested (run with --full)")
    proc = subprocess.run(
        [sys.executable, "evaluation/hardware_benchmark.py"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1800)
    result_path = PROJECT_ROOT / "evaluation" / "results" / "hardware_benchmark.json"
    if proc.returncode != 0 or not result_path.exists():
        return _section("hardware_benchmark", FAIL,
                        (proc.stderr or proc.stdout)[-300:])
    data = json.loads(result_path.read_text(encoding="utf-8"))
    hw = data.get("hardware", {})
    gpu = hw.get("cuda_available", False)
    return _section(
        "hardware_benchmark", PASS,
        "benchmark completed on " + ("GPU" if gpu else "CPU only"),
        gpu_measured=bool(gpu),
        note=None if gpu else
        "GPU acceptance remains NOT VERIFIED: no CUDA GPU on this machine")


def check_camera(requested: bool) -> Dict:
    if not requested:
        return _section("camera", NOT_VERIFIED,
                        "not requested (run with --camera on a machine "
                        "with a webcam)")
    from evaluation.camera_check import run_hardware

    report = run_hardware(with_model=False)
    if report.get("hardware_verified"):
        return _section("camera", PASS,
                        "physical camera session lifecycle verified")
    return _section("camera", NOT_VERIFIED,
                    report.get("result", "no usable camera"))


def check_offline(requested: bool) -> Dict:
    if not requested:
        return _section("offline_mode", NOT_VERIFIED,
                        "not requested (run with --offline)")
    import offline_guard

    offline_guard.enable_offline_mode()
    try:
        import socket

        blocked = False
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=2)
        except OSError:
            blocked = True
        return _section("offline_mode", PASS if blocked else FAIL,
                        "external socket blocked under AEROWATCH_OFFLINE"
                        if blocked else "guard did not block")
    finally:
        offline_guard.disable_offline_mode()


def run_validation(*, full: bool = False, camera: bool = False,
                   offline: bool = False,
                   out_dir: Path | None = None) -> Dict:
    sections: List[Dict] = [
        check_preflight(),
        check_dependencies(),
        check_hardware(),
        check_onnx_smoke(),
        check_database(),
        check_app_startup(),
        check_benchmark(full),
        check_camera(camera),
        check_offline(offline),
    ]
    failures = [s["name"] for s in sections if s["status"] == FAIL]
    unverified = [s["name"] for s in sections if s["status"] == NOT_VERIFIED]
    report = {
        "tool": "release_validation.py (Phase 11)",
        "overall": "FAIL" if failures else "PASS (with NOT VERIFIED items)",
        "failures": failures,
        "not_verified": unverified,
        "sections": sections,
    }
    out = out_dir or (PROJECT_ROOT / "release_validation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2),
                                     encoding="utf-8")
    lines = ["# AeroWatch AI — Release Validation Report", "",
             f"**Overall: {report['overall']}**", "",
             "| Check | Status | Detail |", "|---|---|---|"]
    for s in sections:
        lines.append(f"| {s['name']} | {s['status']} | {s['detail']} |")
    lines += ["", "NOT VERIFIED means the check was not requested or "
              "its hardware was unavailable — it is never a pass."]
    (out / "report.md").write_text("\n".join(lines) + "\n",
                                   encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    report = run_validation(full=args.full, camera=args.camera,
                            offline=args.offline)
    for s in report["sections"]:
        print(f"[{s['status']}] {s['name']}: {s['detail']}")
    print(f"\nOverall: {report['overall']}")
    print("Reports: release_validation/report.json + report.md")
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
