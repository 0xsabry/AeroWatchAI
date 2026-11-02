"""Export the production v7 checkpoint to ONNX (Phase 4).

The .pt original is never modified; its SHA-256 is printed before and
after so any accidental change is visible. The export lands next to
the checkpoint as weights/aerowatch_v7_balanced.onnx.

Usage: python evaluation/export_onnx.py [--weights weights/aerowatch_v7_balanced.pt]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights/aerowatch_v7_balanced.pt")
    args = ap.parse_args()

    weights = (PROJECT_ROOT / args.weights).resolve()
    before = sha256(weights)
    from ultralytics import YOLO

    out = YOLO(str(weights)).export(format="onnx", imgsz=640, dynamic=False, simplify=False, opset=12)
    out_path = Path(out)
    dest = weights.with_suffix(".onnx")
    if out_path.resolve() != dest.resolve():
        shutil.copy(out_path, dest)
    after = sha256(weights)
    assert before == after, "source checkpoint changed during export!"
    print(f"source .pt sha256 (unchanged): {after}")
    print(f"onnx export: {dest} ({dest.stat().st_size} bytes)")
    print(f"onnx sha256: {sha256(dest)}")


if __name__ == "__main__":
    main()
