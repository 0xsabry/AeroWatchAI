"""AeroWatch AI — reproducible hardware benchmark (Phase 4).

Runs the *actual application inference pipeline* (ModelManager predict
+ ai_safety filter path) over the project's test-artifact images and
reports latency / throughput / memory as JSON + a Markdown summary.

Designed for the owner's NVIDIA RTX 3050 4GB machine, but runs
anywhere: it detects the hardware that is actually present and says so.
On a CPU-only machine the GPU sections report "unavailable" — they are
never fabricated.

One command (from the repo root, in the project venv):

    python evaluation/hardware_benchmark.py

Options: --backends torch,onnx  --precisions fp32,fp16  --repeats 3
Results: evaluation/results/hardware_benchmark.json + .md
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def detect_hardware() -> dict:
    hw = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        hw["torch"] = torch.__version__
        hw["cuda_available"] = bool(torch.cuda.is_available())
        if hw["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            hw["gpu_name"] = props.name
            hw["gpu_vram_gb"] = round(props.total_memory / 1e9, 2)
            hw["cuda_version"] = torch.version.cuda
    except Exception as e:
        hw["cuda_available"] = False
        hw["torch_error"] = str(e)
    try:
        import onnxruntime as ort

        hw["onnxruntime"] = ort.__version__
        hw["ort_providers"] = ort.get_available_providers()
    except Exception:
        hw["onnxruntime"] = None
        hw["ort_providers"] = []
    return hw


def load_images() -> list[np.ndarray]:
    from PIL import Image

    paths = sorted((PROJECT_ROOT / "tests" / "artifacts" / "images").glob("*.jpg"))
    paths += sorted((PROJECT_ROOT / "tests" / "artifacts" / "negative_cases").glob("*.jpg"))
    return [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in paths]


def run_config(mm, images: list[np.ndarray], repeats: int) -> dict:
    """Time the full app pipeline: predict + safety filter, per image."""
    # Warm-up (excluded from timings, reported separately)
    t0 = time.perf_counter()
    for img in images[:3]:
        res = mm.model.predict(img, conf=0.25, iou=0.45, imgsz=640, verbose=False)[0]
        mm.filter_airspace_detections(img, res.boxes, 0.25, 0.50)
    warmup_s = time.perf_counter() - t0

    latencies: list[float] = []
    det_counts: list[int] = []
    for _ in range(repeats):
        for img in images:
            start = time.perf_counter()
            res = mm.model.predict(img, conf=0.25, iou=0.45, imgsz=640, verbose=False)[0]
            dets = mm.filter_airspace_detections(img, res.boxes, 0.25, 0.50)
            latencies.append(time.perf_counter() - start)
            det_counts.append(len(dets))
    arr = np.asarray(latencies) * 1000.0
    gpu_mem = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_mem = {
                "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1e6, 1),
            }
    except Exception:
        gpu_mem = None
    return {
        "warmup_seconds_3_images": round(warmup_s, 3),
        "images_per_repeat": len(images),
        "repeats": repeats,
        "latency_ms": {
            "mean": round(float(arr.mean()), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
        },
        "fps_from_mean_latency": round(1000.0 / float(arr.mean()), 2),
        "detections_mean_per_image": round(float(np.mean(det_counts)), 3),
        "gpu_memory": gpu_mem,
        "process_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backends", default="torch,onnx")
    ap.add_argument("--precisions", default="fp32,fp16")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-prefix", default=str(PROJECT_ROOT / "evaluation" / "results" / "hardware_benchmark"))
    args = ap.parse_args()

    from model_manager import ModelManager

    hw = detect_hardware()
    images = load_images()
    results: dict[str, dict] = {}

    def bench(key: str, loader):
        t0 = time.perf_counter()
        mm = loader()
        load_s = time.perf_counter() - t0
        r = run_config(mm, images, args.repeats)
        r["model_load_seconds"] = round(load_s, 3)
        results[key] = r
        print(f"{key}: load={load_s:.2f}s mean={r['latency_ms']['mean']}ms p95={r['latency_ms']['p95']}ms fps={r['fps_from_mean_latency']}")

    if "torch" in args.backends:
        for prec in [p.strip() for p in args.precisions.split(",")]:
            if prec == "fp16" and not hw.get("cuda_available"):
                results["torch_fp16"] = {"skipped": "FP16 requires CUDA; no CUDA device detected"}
                continue
            device = "0" if hw.get("cuda_available") else "cpu"
            key = f"torch_{prec}"
            bench(key, lambda prec=prec, device=device: _load_torch(device, prec == "fp16"))
    if "onnx" in args.backends:
        onnx_path = PROJECT_ROOT / "weights" / "aerowatch_v7_balanced.onnx"
        if onnx_path.exists() and hw.get("onnxruntime"):
            device = "cuda" if "CUDAExecutionProvider" in (hw.get("ort_providers") or []) else "cpu"
            bench(f"onnx_{device}", lambda: _load_onnx(device))
        else:
            results["onnx"] = {"skipped": "ONNX export or onnxruntime not available"}

    payload = {"hardware": hw, "results": results,
               "note": "GPU figures are reported only when a CUDA device was actually detected."}
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    md = ["# Hardware benchmark", "", f"Platform: {hw.get('platform')} · Python {hw.get('python')}",
          f"GPU: {hw.get('gpu_name', 'none detected')} · CUDA available: {hw.get('cuda_available')}", ""]
    for k, r in results.items():
        if "skipped" in r:
            md.append(f"- **{k}**: skipped — {r['skipped']}")
        else:
            md.append(f"- **{k}**: load {r['model_load_seconds']}s · mean {r['latency_ms']['mean']} ms · "
                      f"p50 {r['latency_ms']['p50']} ms · p95 {r['latency_ms']['p95']} ms · "
                      f"{r['fps_from_mean_latency']} FPS · GPU mem {r.get('gpu_memory')} · "
                      f"RSS {r['process_peak_rss_mb']} MB")
    prefix.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"-> {prefix.with_suffix('.json')} / {prefix.with_suffix('.md')}")


def _load_torch(device: str, half: bool):
    from model_manager import ModelManager

    mm = ModelManager()
    mm.load_model(PROJECT_ROOT / "weights" / "aerowatch_v7_balanced.pt",
                  device=device, enable_half=half)
    return mm


def _load_onnx(device: str):
    from model_manager import ModelManager

    mm = ModelManager()
    mm.load_onnx_model(PROJECT_ROOT / "weights" / "aerowatch_v7_balanced.onnx",
                       device=device)
    return mm


if __name__ == "__main__":
    main()
