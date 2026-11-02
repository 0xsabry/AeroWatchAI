"""Phase 5 — fine-tune a candidate on the verified dataset (data/verified_v1).

Base weights: weights/yolo26n.pt — the same clean COCO base v7 started
from (its training data is absent/unverifiable; the candidate does not
inherit it). The verified train split ONLY is used; the frozen test
split is never touched by training, augmentation or tuning.

Negative (background) images: the split's negatives are copied into
train/images with empty label files — standard YOLO background
training for false-positive suppression.

RTX 3050 note: the same script runs on GPU unchanged (device auto).
For a 4 GB card keep batch<=8 at imgsz 640 and amp=True (default on
CUDA). In the Phase 5 sandbox there is no GPU: training ran on CPU
with amp disabled; wall-clock is reported, not extrapolated.

Usage:
  python evaluation/train_verified_candidate.py [--epochs 40] [--smoke]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "verified_v1"


def prep_negatives() -> int:
    n = 0
    for img in sorted((DATA / "train" / "negatives").glob("*.jpg")):
        dest = DATA / "train" / "images" / img.name
        if not dest.exists():
            shutil.copy(img, dest)
            (DATA / "train" / "labels" / (img.stem + ".txt")).write_text("")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true", help="3-epoch pipeline smoke test")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", default="weights/yolo26n.pt")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--name", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    added = prep_negatives()
    print(f"background images added to train: {added}")

    from ultralytics import YOLO

    model = YOLO(str(ROOT / args.base))
    model.train(
        data=str(DATA / "dataset.yaml"),
        epochs=3 if args.smoke else args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=2,
        seed=42,
        lr0=args.lr0,
        deterministic=True,
        patience=12,
        project=str(ROOT / "runs" / "phase5"),
        name="smoke" if args.smoke else (args.name or "verified_v1_ft"),
        exist_ok=True,
        plots=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
