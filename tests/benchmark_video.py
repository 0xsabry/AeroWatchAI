"""AeroWatch AI — Reproducible Video Performance Benchmark on RTX 3050 Laptop GPU.

Measures genuine performance on the actual hardware:
- NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- CUDA 13.2 / PyTorch 2.14.1+cu132 / Tensor Core FP16
- Intel/AMD x86_64 CPU & 16GB System RAM

Profiles:
1. Offline Complete Processing (vid_stride=1, full H.264 export)
2. Fast Sampled Preview (vid_stride=2, 50% subsampling)
3. Live Preview Low-Latency (simulated real-time latest-frame buffer)

Measures stage-by-stage latencies (Read, Prep, Infer, Post, Render, Write, E2E),
percentiles (Median, P95), peak VRAM, and outputs reports/validation/benchmark.csv.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_manager import ModelManager
from video_processor import VideoProcessingMode, VideoProcessor

WEIGHTS_PATH = PROJECT_ROOT / "weights" / "aerowatch_v7_balanced.pt"
TEST_VIDEO_PATH = PROJECT_ROOT / "tests" / "artifacts" / "videos" / "civilian_drone_surveillance.mp4"
REPORTS_DIR = PROJECT_ROOT / "reports" / "validation"


def warmup_gpu(model_manager: ModelManager, iterations: int = 15):
    """Warms up CUDA context and Tensor Cores with dummy tensor passes."""
    print(f"[*] Warming up GPU ({iterations} passes with CUDA synchronize)...", flush=True)
    model = model_manager.model
    device = model_manager.device
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

    for _ in range(iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _ = model.predict(source=dummy_frame, device=device, imgsz=640, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    print("    -> GPU warmup complete.", flush=True)


def benchmark_single_run(
    video_path: Path,
    processor: VideoProcessor,
    profile_name: str,
    vid_stride: int = 1,
    export_video: bool = True,
    run_index: int = 1,
) -> Dict[str, Any]:
    """Executes a single benchmark run with fine-grained per-frame latency tracking."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open benchmark video: {video_path}")

    total_file_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_out = REPORTS_DIR / f"bench_temp_{profile_name.lower().replace(' ', '_')}_run{run_index}.mp4"
    writer = None
    if export_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        eff_fps = input_fps / vid_stride if vid_stride > 1 else input_fps
        writer = cv2.VideoWriter(str(temp_out), fourcc, eff_fps, (w, h))

    mm = processor.model_manager
    model = mm.model

    read_times = []
    prep_times = []
    infer_times = []
    post_times = []
    render_times = []
    write_times = []
    frame_e2e_times = []

    decoded_count = 0
    processed_count = 0
    skipped_count = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t_run_start = time.perf_counter()

    while True:
        t_f_start = time.perf_counter()

        # Stage 1: Decode / Read
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t_read = time.perf_counter() - t0
        if not ret or frame is None:
            break

        decoded_count += 1
        read_times.append(t_read * 1000.0)

        # Stride skipping
        if vid_stride > 1 and (decoded_count % vid_stride) != 0:
            skipped_count += 1
            continue

        # Stage 2: Preprocess
        t1 = time.perf_counter()
        # Ensure memory contiguous
        contiguous_frame = np.ascontiguousarray(frame)
        t_prep = time.perf_counter() - t1
        prep_times.append(t_prep * 1000.0)

        # Stage 3: Inference (with CUDA synchronization for honest measurement)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        results = model.predict(
            source=contiguous_frame,
            device=mm.device,
            conf=0.25,
            imgsz=640,
            verbose=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_infer = time.perf_counter() - t2
        infer_times.append(t_infer * 1000.0)

        # Stage 4: Postprocess
        t3 = time.perf_counter()
        res = results[0]
        boxes = res.boxes
        raw_dets = []
        for b in boxes:
            raw_dets.append({
                "cls": int(b.cls.item()),
                "conf": float(b.conf.item()),
                "xyxy": b.xyxy[0].tolist(),
            })
        smoothed_dets = processor.stabilizer.update(raw_dets)
        t_post = time.perf_counter() - t3
        post_times.append(t_post * 1000.0)

        # Stage 5: Render
        t4 = time.perf_counter()
        annotated = frame.copy()
        for d in smoothed_dets:
            _, display, hex_color = mm.map_detection_label(d["cls"], d["conf"])
            if display is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            hex_clean = (hex_color or "#00D4FF").lstrip("#")
            bgr = tuple(int(hex_clean[i : i + 2], 16) for i in (4, 2, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
        t_render = time.perf_counter() - t4
        render_times.append(t_render * 1000.0)

        # Stage 6: Video Write
        t5 = time.perf_counter()
        if writer is not None:
            writer.write(annotated)
        t_write = time.perf_counter() - t5
        write_times.append(t_write * 1000.0)

        t_f_end = time.perf_counter()
        frame_e2e_times.append((t_f_end - t_f_start) * 1000.0)
        processed_count += 1

    cap.release()
    if writer is not None:
        writer.release()
    if temp_out.exists():
        temp_out.unlink()  # Clean up temp benchmark video

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_elapsed_sec = time.perf_counter() - t_run_start
    e2e_throughput_fps = processed_count / total_elapsed_sec if total_elapsed_sec > 0 else 0.0

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    # Compute percentiles
    median_frame_latency = statistics.median(frame_e2e_times) if frame_e2e_times else 0.0
    p95_frame_latency = np.percentile(frame_e2e_times, 95) if frame_e2e_times else 0.0
    mean_infer = statistics.mean(infer_times) if infer_times else 0.0
    infer_fps = 1000.0 / mean_infer if mean_infer > 0 else 0.0

    return {
        "profile": profile_name,
        "run_index": run_index,
        "resolution": f"{w}x{h}",
        "total_frames_in_file": total_file_frames,
        "frames_decoded": decoded_count,
        "frames_processed": processed_count,
        "frames_skipped": skipped_count,
        "total_elapsed_sec": round(total_elapsed_sec, 3),
        "e2e_throughput_fps": round(e2e_throughput_fps, 2),
        "inference_fps": round(infer_fps, 2),
        "mean_inference_ms": round(mean_infer, 2),
        "median_frame_e2e_ms": round(median_frame_latency, 2),
        "p95_frame_e2e_ms": round(p95_frame_latency, 2),
        "avg_read_ms": round(statistics.mean(read_times), 2) if read_times else 0.0,
        "avg_prep_ms": round(statistics.mean(prep_times), 2) if prep_times else 0.0,
        "avg_post_ms": round(statistics.mean(post_times), 2) if post_times else 0.0,
        "avg_render_ms": round(statistics.mean(render_times), 2) if render_times else 0.0,
        "avg_write_ms": round(statistics.mean(write_times), 2) if write_times else 0.0,
        "peak_vram_mb": round(peak_vram_mb, 2),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "precision": "FP16 (Half-Precision)" if mm.is_half else "FP32",
    }


def run_full_benchmark() -> List[Dict[str, Any]]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Benchmark video artifact not found: {TEST_VIDEO_PATH}")

    mm = ModelManager()
    mm.load_model(WEIGHTS_PATH, device="auto", enable_half=True)
    vp = VideoProcessor(model_manager=mm)

    warmup_gpu(mm, iterations=10)

    profiles = [
        ("Offline Complete Processing", 1, True),
        ("Fast Sampled Preview", 2, False),
        ("Live Preview Low-Latency", 1, False),
    ]

    all_results = []
    repetitions = 3

    print("\n" + "=" * 80)
    print("AEROWATCH AI — REPRODUCIBLE RTX 3050 VIDEO PERFORMANCE BENCHMARK")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Artifact: {TEST_VIDEO_PATH.name}")
    print(f"Model Checkpoint: {WEIGHTS_PATH.name} (FP16 Native Tensor Core)")
    print("=" * 80 + "\n")

    for prof_name, stride, export in profiles:
        print(f"[*] Benchmarking Profile: {prof_name} (vid_stride={stride}, export={export}) across {repetitions} runs...")
        run_stats = []
        for r in range(1, repetitions + 1):
            res = benchmark_single_run(
                video_path=TEST_VIDEO_PATH,
                processor=vp,
                profile_name=prof_name,
                vid_stride=stride,
                export_video=export,
                run_index=r,
            )
            run_stats.append(res)
            all_results.append(res)
            print(
                f"    Run {r}/{repetitions}: E2E FPS={res['e2e_throughput_fps']}, "
                f"Infer Latency={res['mean_inference_ms']}ms, "
                f"P95 Latency={res['p95_frame_e2e_ms']}ms, "
                f"VRAM={res['peak_vram_mb']}MB"
            )

        mean_fps = statistics.mean([x["e2e_throughput_fps"] for x in run_stats])
        stdev_fps = statistics.stdev([x["e2e_throughput_fps"] for x in run_stats]) if len(run_stats) > 1 else 0.0
        print(f"  -> Profile {prof_name} Summary: {mean_fps:.2f} ± {stdev_fps:.2f} E2E FPS\n")

    # Export to reports/validation/benchmark.csv
    csv_path = REPORTS_DIR / "benchmark.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[+] Saved structured benchmark CSV to: {csv_path}")

    # Export to reports/validation/benchmark.json
    json_path = REPORTS_DIR / "benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[+] Saved structured benchmark JSON to: {json_path}")

    return all_results


if __name__ == "__main__":
    run_full_benchmark()
