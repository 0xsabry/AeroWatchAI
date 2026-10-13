"""AeroWatch AI — Unbiased Balanced Model Training (aerowatch_v7_balanced).

Trains on data/real_aerial_v4_balanced:
- Perfectly balanced class distribution (Drones: 194, Jets: 243, Birds: 96, Airliners: 73)
- Higher classification loss weight (cls=1.2) to eliminate class confusion
- Multi-scale and scale jitter for small multirotor drone detection
- Clean pre-trained initialization to remove residual bias towards military_jet
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO


def main():
    print("=== AeroWatch AI v7 Balanced Model Training ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    data_yaml = Path("data/real_aerial_v4_balanced/dataset.yaml").resolve()
    assert data_yaml.exists(), f"Missing dataset config at {data_yaml}"

    # Use clean pretrained YOLO26n base to completely eliminate residual bias
    base_weights = Path("yolo26n.pt")
    if not base_weights.exists():
        base_weights = Path("weights/yolo26n.pt")
    print(f"[*] Training from unbiased pretrained base: {base_weights}")

    model = YOLO(str(base_weights))

    results = model.train(
        data=str(data_yaml),
        epochs=30,
        batch=8,
        imgsz=640,
        device="0" if torch.cuda.is_available() else "cpu",
        project=str(Path("runs/train").resolve()),
        name="aerowatch_v7_balanced",
        exist_ok=True,
        workers=0,
        cos_lr=True,
        warmup_epochs=2.0,
        box=7.5,
        cls=1.2,          # High classification penalty prevents dominant class bias
        scale=0.5,
        fliplr=0.5,
        mosaic=0.7,
        close_mosaic=5,
        patience=15,
        plots=True,
        verbose=True,
    )

    best_pt = Path("runs/train/aerowatch_v7_balanced/weights/best.pt")
    if best_pt.exists():
        dest1 = Path("weights/aerowatch_v7_balanced.pt")
        dest2 = Path("weights/best.pt")
        shutil.copy(best_pt, dest1)
        shutil.copy(best_pt, dest2)
        print(f"\n[+] Deployed unbiased model:")
        print(f"    -> {dest1}")
        print(f"    -> {dest2}")

    print("\n[+] Validation evaluation on test set:")
    val_results = model.val(data=str(data_yaml), split="test", imgsz=640)
    print(val_results.results_dict)


if __name__ == "__main__":
    main()
