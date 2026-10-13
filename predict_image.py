"""AeroWatch AI — Civilian Aerial Object Detection CLI.

Uses ModelManager to provide canonical civilian vocabulary:
  - drone    -> Drone
  - airplane -> Aircraft
  - bird     -> Bird
  - balloon  -> Balloon
Supports CUDA FP16 Tensor Core acceleration and saves annotated visual results to runs/predict/<name>.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from model_manager import ModelManager


def parse_arguments():
    parser = argparse.ArgumentParser(description="AeroWatch AI Image Prediction")
    parser.add_argument("--source", required=True, help="Path to input image file")
    parser.add_argument("--weights", default="weights/aerowatch_v7_balanced.pt", help="Path to model weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence detection threshold (0.0 to 1.0)")
    parser.add_argument("--uncertain", type=float, default=0.40, help="Confidence threshold below which detections are flagged Uncertain")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--device", default="auto", help="Compute device ('auto', '0', 'cpu')")
    parser.add_argument("--project", default="runs/predict", help="Output directory root")
    parser.add_argument("--name", default="exp", help="Inference experiment subfolder")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO label text file")
    return parser.parse_args()


def predict_image(args):
    img_path = Path(args.source).resolve()
    if not img_path.is_file():
        raise SystemExit(f"[ERROR] Input image not found: {img_path}")

    weights_path = Path(args.weights).resolve()
    if not weights_path.is_file():
        alt = Path("weights") / args.weights
        if alt.is_file():
            weights_path = alt.resolve()
        else:
            raise SystemExit(f"[ERROR] Checkpoint weights not found: {weights_path}")

    project_dir = Path(args.project).resolve()
    out_folder = project_dir / args.name
    out_folder.mkdir(parents=True, exist_ok=True)

    mm = ModelManager()
    meta = mm.introspect_checkpoint(weights_path)
    model = mm.load_model(weights_path, device=args.device, imgsz=args.imgsz)

    print("=" * 60)
    print("AEROWATCH AI — CIVILIAN AERIAL OBJECT INFERENCE")
    print(f"Input image       : {img_path.name}")
    print(f"Model Checkpoint  : {weights_path.name} ({meta.model_type.value})")
    print(f"Compute Hardware  : {mm.device} (FP16: {mm.is_half})")
    print(f"Thresholds        : Detection={args.conf}, Uncertain={args.uncertain}")
    print(f"Supports Drones   : {meta.supports_drones()}")
    print("=" * 60)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise SystemExit(f"[ERROR] OpenCV could not read image: {img_path}")

    results = model.predict(
        source=img_bgr,
        imgsz=args.imgsz,
        conf=args.conf,
        device=mm.device,
        verbose=False,
    )

    res = results[0]
    boxes = res.boxes
    annotated = img_bgr.copy()

    print("\n--- CANONICAL DETECTION RESULTS ---")
    detected_count = 0
    txt_lines = []

    h, w = img_bgr.shape[:2]

    for i, b in enumerate(boxes, start=1):
        cid = int(b.cls.item())
        conf = float(b.conf.item())
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]

        internal_name, display_label, hex_color = mm.map_detection_label(
            cid, conf, args.conf, args.uncertain
        )
        if display_label is None:
            continue

        detected_count += 1
        hex_clean = (hex_color or "#00D4FF").lstrip("#")
        bgr_col = tuple(int(hex_clean[idx : idx + 2], 16) for idx in (4, 2, 0))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_col, 2)
        lbl_text = f"{display_label}: {conf:.1%}"
        (tw, th), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), bgr_col, -1)
        cv2.putText(annotated, lbl_text, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1)

        print(f"  [{detected_count}] {display_label:18s} | Confidence: {conf:.1%} | BBox: [{x1}, {y1}, {x2}, {y2}]")

        if args.save_txt:
            xc = ((x1 + x2) / 2.0) / w
            yc = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            txt_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.4f}\n")

    if detected_count == 0:
        print("  (No aerial objects detected above confidence threshold)")

    out_file = out_folder / img_path.name
    cv2.imwrite(str(out_file), annotated)

    if args.save_txt and txt_lines:
        txt_file = out_folder / f"{img_path.stem}.txt"
        txt_file.write_text("".join(txt_lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Saved annotated image to: {out_file}")
    print("=" * 60)


def main():
    args = parse_arguments()
    predict_image(args)


if __name__ == "__main__":
    main()
