"""AeroWatch AI — YOLO26n Training Pipeline for Civilian Aerial Object Recognition.

Runs strict pre-flight dataset validation before initiating Ultralytics training.
Optimized for NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) on Windows.

Outputs clean directory structure:
  runs/train/<run_name>/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from validate_dataset import validate_dataset, load_dataset_yaml, CANONICAL_CLASSES


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validated YOLO26n Training for AeroWatch AI")
    parser.add_argument("--data", default="data/synthetic_v3/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained weights checkpoint path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (reduce to 4 or 2 if VRAM is tight)")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size (512 or 640)")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu", help="Compute device ('0' for CUDA, or 'cpu')")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 recommended on Windows)")
    parser.add_argument("--project", default="runs/train", help="Training root directory")
    parser.add_argument("--name", default="aerowatch_v3", help="Specific experiment name")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--min-train", type=int, default=100, help="Minimum required training images")
    parser.add_argument("--min-val", type=int, default=20, help="Minimum required validation images")
    parser.add_argument("--smoke-test", action="store_true", help="Run an explicit 1-epoch technical smoke test (metrics are NOT meaningful)")
    return parser.parse_args()


def run_training(args):
    yaml_path = Path(args.data).resolve()
    if not yaml_path.is_file():
        raise SystemExit(f"[FATAL] Dataset configuration not found: {yaml_path}")

    weights_path = Path(args.model).resolve()
    if not weights_path.is_file():
        # Fallback to local weights folder if available
        alt_path = Path("weights") / args.model
        if alt_path.is_file():
            weights_path = alt_path.resolve()
        else:
            raise SystemExit(f"[FATAL] Model weights not found: {weights_path}")

    print("=" * 65)
    print("AEROWATCH AI — TRAINING PRE-FLIGHT CHECKS")
    print(f"Dataset config : {yaml_path}")
    print(f"Initial model  : {weights_path}")
    print(f"Device         : {args.device} (CUDA available: {torch.cuda.is_available()})")
    if torch.cuda.is_available() and args.device != "cpu":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Hardware   : {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"Hyperparams    : epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, workers={args.workers}")
    print("=" * 65)

    # 1. MANDATORY VALIDATION SAFEGUARDS
    print("\n[Step 1/3] Executing strict dataset validation...")
    val_args = argparse.Namespace(
        data=str(yaml_path),
        min_train=args.min_train,
        min_val=args.min_val,
        min_test=10,
        draft_mode=args.smoke_test,
        strict_names=True,
    )

    is_valid = validate_dataset(val_args)
    if not is_valid:
        print("\n[STOP] Training refused. The dataset failed strict pre-flight validation.")
        print("Safeguards prevent training on corrupted, invalid, or undersized datasets.")
        sys.exit(1)

    # If smoke test is requested, override epochs to 1 for quick execution
    if args.smoke_test:
        print("\n" + "!" * 65)
        print("WARNING: --smoke-test mode is ACTIVE.")
        print("This run will train for 1 epoch only to verify technical GPU execution.")
        print("The resulting model weights will NOT be accurate or useful for production inference.")
        print("!" * 65)
        epochs = 1
        run_name = f"{args.name}_smoke"
    else:
        epochs = args.epochs
        run_name = args.name

    # 2. OUTPUT PATH CONFIGURATION (Avoids nested directories)
    # Using absolute resolved path for project ensures Ultralytics writes directly to:
    # <workspace>/runs/train/<run_name>/...
    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Step 2/3] Output directory: {project_dir / run_name}")

    # 3. INITIALIZE AND TRAIN MODEL
    print(f"\n[Step 3/3] Loading {weights_path.name} and starting YOLO training...")
    model = YOLO(str(weights_path))

    training_start = time.time()
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            amp=True,
            cache=False,
            patience=args.patience,
            seed=42,
            deterministic=True,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            plots=True,
            save=True,
            val=True,
            verbose=True,
        )
    except torch.cuda.OutOfMemoryError:
        print("\n[FATAL ERROR] CUDA Out of Memory!")
        print("Suggestions for 4 GB VRAM:")
        print("  - Reduce batch size to 4 or 2: --batch 4")
        print("  - Reduce image size to 512: --imgsz 512")
        sys.exit(2)

    total_training_time = round(time.time() - training_start, 2)
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else project_dir / run_name

    print("\n" + "=" * 65)
    print("TRAINING RUN COMPLETE")
    print(f"Run directory   : {save_dir}")
    print(f"Total time      : {total_training_time} seconds")
    print(f"Best weights    : {save_dir / 'weights' / 'best.pt'}")
    print(f"Last weights    : {save_dir / 'weights' / 'last.pt'}")

    if args.smoke_test:
        print("\n[REMINDER] This was a technical smoke test. Do NOT treat these metrics as real performance.")
    else:
        print(f"Evaluation metrics and curves saved to: {save_dir / 'results.csv'}")
    print("=" * 65)


def main():
    args = parse_arguments()
    run_training(args)


if __name__ == "__main__":
    main()
