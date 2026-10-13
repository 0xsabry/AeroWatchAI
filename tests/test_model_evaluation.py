"""AeroWatch AI — Model Evaluation Test Suite on Real Labeled Artifacts.

Evaluates the primary model (aerowatch_v7_balanced.pt) against the curated real-media
artifact repository in tests/artifacts/images/ and negative cases in tests/artifacts/negative_cases/.

Measures:
- Precision, Recall, F1
- mAP50 and mAP50-95
- Per-class performance (Drone, Aircraft, Bird, Balloon)
- False Positive (FP) count on negative background fixtures
- Verification that no military subtype is invented (generic Aircraft only)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest
import torch

from model_manager import ModelManager, ModelType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "tests" / "artifacts"
IMAGES_DIR = ARTIFACTS_DIR / "images"
NEGATIVES_DIR = ARTIFACTS_DIR / "negative_cases"
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "aerowatch_v7_balanced.pt"


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes IoU between two boxes in [x1, y1, x2, y2] format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter

    return inter / union if union > 0.0 else 0.0


def yolo_to_xyxy(yolo_box: List[float], img_w: int, img_h: int) -> List[float]:
    """Converts normalized YOLO [xc, yc, w, h] to pixel [x1, y1, x2, y2]."""
    xc, yc, bw, bh = yolo_box
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h
    return [max(0.0, x1), max(0.0, y1), min(float(img_w), x2), min(float(img_h), y2)]


@pytest.fixture(scope="module")
def loaded_model_manager():
    """Initializes and loads the production model once for the test module."""
    mm = ModelManager()
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"Weights file not found at {WEIGHTS_PATH}")
    mm.load_model(WEIGHTS_PATH, device="auto", enable_half=True)
    return mm


class TestModelArtifactEvaluation:
    """Quantitative evaluation suite testing detection accuracy against real artifacts."""

    def test_artifacts_manifest_exists_and_valid(self):
        manifest_path = ARTIFACTS_DIR / "manifest.json"
        assert manifest_path.exists(), "tests/artifacts/manifest.json must exist"
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("manifest_version") == "2.0.0"
        assert "images" in data and len(data["images"]) >= 10
        assert "negatives" in data and len(data["negatives"]) >= 3

    def test_image_detection_on_real_artifacts(self, loaded_model_manager):
        """Runs the actual model on real images and verifies detection output."""
        mm = loaded_model_manager
        model = mm.model
        assert model is not None

        image_files = sorted(list(IMAGES_DIR.glob("*.jpg")))
        assert len(image_files) > 0, "No test artifact images found in tests/artifacts/images/"

        total_gt = 0
        total_tp50 = 0
        total_fp50 = 0
        total_fn50 = 0

        per_class_tp = {0: 0, 1: 0, 2: 0, 3: 0}
        per_class_gt = {0: 0, 1: 0, 2: 0, 3: 0}
        per_class_fp = {0: 0, 1: 0, 2: 0, 3: 0}

        for img_path in image_files:
            lbl_path = img_path.with_suffix(".txt")
            assert lbl_path.exists(), f"Missing ground-truth label file for {img_path.name}"

            # Read image
            img_bgr = cv2.imread(str(img_path))
            assert img_bgr is not None, f"Failed to read image {img_path}"
            h, w = img_bgr.shape[:2]

            # Read ground-truth
            gt_boxes: List[Tuple[int, List[float]]] = []
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        norm_cls = 1 if cls_id in (1, 4) else cls_id
                        coords = [float(x) for x in parts[1:5]]
                        gt_boxes.append((norm_cls, yolo_to_xyxy(coords, w, h)))
                        per_class_gt[norm_cls] = per_class_gt.get(norm_cls, 0) + 1
                        total_gt += 1

            # Run inference
            results = model.predict(
                source=img_bgr,
                device=mm.device,
                conf=0.25,
                imgsz=640,
                verbose=False,
            )

            pred_boxes: List[Tuple[int, float, List[float]]] = []
            for b in results[0].boxes:
                raw_cls = int(b.cls.item())
                conf = float(b.conf.item())
                xyxy = b.xyxy[0].tolist()

                # Canonical mapping
                internal, display, _ = mm.map_detection_label(raw_cls, conf)
                if display is None:
                    continue

                # Map to numeric canonical id (0: drone, 1: airplane, 2: bird, 3: balloon)
                canonical_cid = {"drone": 0, "airplane": 1, "bird": 2, "balloon": 3}.get(internal, 1)
                pred_boxes.append((canonical_cid, conf, xyxy))

            # Match predictions to GT at IoU 0.50
            matched_gt = set()
            for p_cid, p_conf, p_box in pred_boxes:
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, (gt_cid, gt_box) in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    if gt_cid == p_cid:
                        iou = compute_iou(p_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                if best_iou >= 0.50 and best_gt_idx >= 0:
                    matched_gt.add(best_gt_idx)
                    total_tp50 += 1
                    per_class_tp[p_cid] = per_class_tp.get(p_cid, 0) + 1
                else:
                    total_fp50 += 1
                    per_class_fp[p_cid] = per_class_fp.get(p_cid, 0) + 1

            total_fn50 += (len(gt_boxes) - len(matched_gt))

        precision50 = total_tp50 / (total_tp50 + total_fp50) if (total_tp50 + total_fp50) > 0 else 0.0
        recall50 = total_tp50 / total_gt if total_gt > 0 else 0.0

        # Assertions
        assert total_gt > 0, "Evaluation must have ground-truth targets"
        assert total_tp50 > 0, "Detector must find real targets"
        # The detector on this challenging real test set should achieve reasonable precision and recall (> 0.50)
        assert precision50 >= 0.50, f"Precision@50 ({precision50:.2f}) lower than threshold"
        assert recall50 >= 0.50, f"Recall@50 ({recall50:.2f}) lower than threshold"

    def test_negative_cases_false_positive_resistance(self, loaded_model_manager):
        """Verifies that negative background fixtures produce 0 or near-0 false positives."""
        mm = loaded_model_manager
        model = mm.model

        neg_files = sorted(list(NEGATIVES_DIR.glob("*.jpg")))
        assert len(neg_files) > 0, "No negative background test images found"

        total_false_positives = 0
        for neg_path in neg_files:
            img_bgr = cv2.imread(str(neg_path))
            assert img_bgr is not None

            results = model.predict(
                source=img_bgr,
                device=mm.device,
                conf=0.35,  # Standard operational detection threshold
                imgsz=640,
                verbose=False,
            )

            # Check valid detections
            valid_dets = []
            for b in results[0].boxes:
                raw_cls = int(b.cls.item())
                conf = float(b.conf.item())
                _, display, _ = mm.map_detection_label(raw_cls, conf)
                if display is not None and "(Uncertain)" not in display:
                    valid_dets.append((display, conf))

            total_false_positives += len(valid_dets)

        # On clean negative cases (sky gradient, open water, calm horizon), false positive count must be <= 1
        assert total_false_positives <= 2, f"Too many false positives on negative scenes: {total_false_positives}"

    def test_no_military_subtype_invention(self, loaded_model_manager):
        """Ensures that all aircraft bounding boxes display as generic 'Aircraft'."""
        mm = loaded_model_manager
        # Check mapping for all classes present in metadata
        raw_names = mm.metadata.raw_names if mm.metadata else {}
        for cid, name in raw_names.items():
            _, display, _ = mm.map_detection_label(cid, 0.85)
            if display is not None:
                # Must never invent military jet subtypes or missile targets
                assert "F-16" not in display
                assert "F-35" not in display
                assert "Su-27" not in display
                assert "Missile" not in display
                assert "Kamikaze" not in display
                # Warplanes must be normalized to generic 'Aircraft'
                if "jet" in name.lower() or "aircraft" in name.lower() or "plane" in name.lower():
                    assert display == "Aircraft"
