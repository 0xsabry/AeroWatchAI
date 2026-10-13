"""Optional baseline fine-tuning on the synthetic educational dataset.

Run only after generating enough labeled images and reviewing the labels:
  python train_yolo.py

This is a baseline experiment, not evidence of operational reliability.
"""
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data" / "synthetic_v1" / "dataset.yaml"

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this Python environment. Check the PyTorch install first.")
if not DATA_YAML.exists():
    raise SystemExit(f"Dataset config not found: {DATA_YAML}. Run the Blender generator first.")

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

# Nano is a deliberate starting point for a laptop GPU with 4 GB VRAM.
model = YOLO("yolo26n.pt")
model.train(
    data=str(DATA_YAML),
    imgsz=640,
    epochs=30,
    batch=2,
    device=0,
    workers=0,
    patience=8,
    cache=False,
    amp=True,
    project=str(ROOT / "runs" / "detect"),
    name="synthetic_baseline",
    exist_ok=True,
)
