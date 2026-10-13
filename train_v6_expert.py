"""AeroWatch AI — High-Accuracy Model Training (aerowatch_v6_expert).

Trains YOLO26n on data/real_aerial_v3 with:
- Balanced real-world dataset (Commercial airliners, UAVs, warplanes, birds, balloons, hard negatives)
- Cosine annealing learning rate schedule (cos_lr=True)
- Multi-scale and scale jitter for small drone detection at distance
- Fine-tuned box and classification loss weighting
- CUDA acceleration on NVIDIA RTX 3050 Laptop GPU
"""

from __future__ import annotations

import os
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    print("=== AeroWatch AI v6 Expert Model Training ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    data_yaml = Path("data/real_aerial_v3/dataset.yaml").resolve()
    assert data_yaml.exists(), f"Missing dataset config at {data_yaml}"

    # Initialize from the proven aerowatch_v5_pro weights
    base_weights = Path("weights/aerowatch_v5_pro.pt")
    if not base_weights.exists():
        base_weights = Path("weights/aerowatch_real_v1.pt")
    print(f"[*] Starting transfer learning from: {base_weights}")

    model = YOLO(str(base_weights))

    results = model.train(
        data=str(data_yaml),
        epochs=30,
        batch=8,
        imgsz=640,
        device="0" if torch.cuda.is_available() else "cpu",
        project=str(Path("runs/train").resolve()),
        name="aerowatch_v6_expert",
        exist_ok=True,
        workers=0,
        cos_lr=True,
        warmup_epochs=2.0,
        box=7.5,
        cls=0.7,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.7,
        close_mosaic=5,
        patience=15,
        plots=True,
        verbose=True,
    )

    best_pt = Path("runs/train/aerowatch_v6_expert/weights/best.pt")
    if best_pt.exists():
        dest1 = Path("weights/aerowatch_v6_expert.pt")
        dest2 = Path("weights/best.pt")
        import shutil
        shutil.copy(best_pt, dest1)
        shutil.copy(best_pt, dest2)
        print(f"\n[+] Trained model deployed successfully:")
        print(f"    -> {dest1}")
        print(f"    -> {dest2}")

    print("\n[+] Validation evaluation on test set:")
    val_results = model.val(data=str(data_yaml), split="test", imgsz=640)
    print(val_results.results_dict)


if __name__ == "__main__":
    main()
