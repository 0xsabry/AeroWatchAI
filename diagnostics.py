"""AeroWatch AI — System Diagnostics & Hardware Verification Suite.

Runs automated system-level checks across:
- Compute hardware (CUDA, GPU model, VRAM limits, PyTorch build)
- Windows camera subsystems (DirectShow & Media Foundation probing)
- Video decoding and H.264 encoding support (OpenCV, FFmpeg)
- Checkpoint integrity and metadata introspection
- Dataset directory structure and canonical YAML mappings
"""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import torch
from ultralytics import YOLO

from camera_manager import CameraManager
from model_manager import ModelManager, ModelType, discover_checkpoints


@dataclass
class SystemDiagnosticReport:
    os_name: str
    python_version: str
    pytorch_version: str
    cuda_available: bool
    gpu_name: str
    vram_total_gb: float
    vram_free_gb: float
    opencv_version: str
    ffmpeg_available: bool
    available_cameras: List[Dict[str, Any]]
    available_models: List[Dict[str, Any]]
    dataset_summary: List[Dict[str, Any]]
    warnings_or_errors: List[str]


def run_full_diagnostics(project_root: Path) -> SystemDiagnosticReport:
    """Executes non-destructive end-to-end diagnostics and returns a structured report."""
    warnings: List[str] = []

    # 1. Hardware & Platform
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
    vram_total = 0.0
    vram_free = 0.0

    if cuda_avail:
        try:
            total_b = torch.cuda.get_device_properties(0).total_memory
            vram_total = round(total_b / (1024**3), 2)
            free_b, _ = torch.cuda.mem_get_info()
            vram_free = round(free_b / (1024**3), 2)
        except Exception as e:
            warnings.append(f"Could not query exact GPU VRAM info: {e}")
    else:
        warnings.append("CUDA is NOT available. Model will run in slow CPU mode.")

    # 2. FFmpeg / Codec Verification
    ffmpeg_ok = False
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            ffmpeg_ok = True
    except Exception:
        warnings.append("imageio_ffmpeg is not available. Browser-compatible H.264 encoding may fail.")

    # 3. Camera Diagnostics
    camera_probes = []
    try:
        raw_cams = CameraManager.probe_available_cameras(max_indices=3)
        for c in raw_cams:
            camera_probes.append(
                {
                    "index": c.index,
                    "backend": c.backend.name,
                    "available": c.is_available,
                    "can_read": c.can_read,
                    "resolution": f"{c.width}x{c.height}" if c.can_read else "N/A",
                    "error": c.error_message,
                }
            )
        if not any(c.can_read for c in raw_cams):
            warnings.append("No readable webcam found on this computer. Live camera mode will be unavailable.")
    except Exception as e:
        warnings.append(f"Camera probing threw an exception: {e}")

    # 4. Checkpoint Diagnostics
    model_entries = []
    mm = ModelManager()
    found_pts = discover_checkpoints(project_root)
    for label, path, m_type in found_pts:
        try:
            meta = mm.introspect_checkpoint(path)
            model_entries.append(
                {
                    "label": label,
                    "path": str(path.relative_to(project_root) if project_root in path.parents else path.name),
                    "model_type": meta.model_type.value,
                    "num_classes": meta.num_classes,
                    "supports_drones": meta.supports_drones(),
                    "supports_balloons": meta.supports_balloons(),
                    "raw_classes": list(meta.raw_names.values())[:6],
                }
            )
        except Exception as e:
            warnings.append(f"Error reading checkpoint {path.name}: {e}")

    # 5. Dataset Diagnostics
    dataset_entries = []
    data_dir = project_root / "data"
    if data_dir.exists():
        for d in sorted(data_dir.iterdir()):
            if d.is_dir():
                yaml_p = d / "dataset.yaml"
                if not yaml_p.exists():
                    yaml_p = d / "data.yaml"

                has_yaml = yaml_p.exists()
                train_count = len(list((d / "images" / "train").glob("*.*"))) if (d / "images" / "train").exists() else 0
                val_count = len(list((d / "images" / "val").glob("*.*"))) if (d / "images" / "val").exists() else 0
                test_count = len(list((d / "images" / "test").glob("*.*"))) if (d / "images" / "test").exists() else 0

                dataset_entries.append(
                    {
                        "name": d.name,
                        "has_yaml": has_yaml,
                        "train_images": train_count,
                        "val_images": val_count,
                        "test_images": test_count,
                    }
                )

    return SystemDiagnosticReport(
        os_name=f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        cuda_available=cuda_avail,
        gpu_name=gpu_name,
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
        opencv_version=cv2.__version__,
        ffmpeg_available=ffmpeg_ok,
        available_cameras=camera_probes,
        available_models=model_entries,
        dataset_summary=dataset_entries,
        warnings_or_errors=warnings,
    )
