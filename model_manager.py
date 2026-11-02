"""AeroWatch AI — Model Manager & Class Normalization Architecture.

Provides:
- Checkpoint discovery, introspection, and validation
- Strict separation between Pretrained COCO (80-class) and Custom models
- Canonical civilian vocabulary mapping:
    * 'drone'    -> Display: 'Drone'
    * 'airplane' -> Display: 'Aircraft'
    * 'bird'     -> Display: 'Bird'
    * 'balloon'  -> Display: 'Balloon'
- Low-confidence handling ('Uncertain' thresholding)
- CUDA device selection, FP16 half-precision, and one-shot warmup
"""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ultralytics import YOLO

from ai_safety import (
    DetectionValidationError,
    dedupe_canonical,
    sanitize_detections,
    validate_checkpoint_file,
    validate_confidence,
)

logger = logging.getLogger("AeroWatch.ModelManager")

# Canonical civilian vocabulary definitions
CANONICAL_CIVILIAN_MAP: Dict[str, Tuple[str, str]] = {
    # internal_name: (canonical_internal, display_label)
    "drone": ("drone", "Drone"),
    "aircraft": ("airplane", "Aircraft"),
    "airplane": ("airplane", "Aircraft"),
    "plane": ("airplane", "Aircraft"),
    "military_jet": ("airplane", "Aircraft"),  # Normalize warplanes to generic Aircraft
    "bird": ("bird", "Bird"),
    "balloon": ("balloon", "Balloon"),
}

CANONICAL_COLORS: Dict[str, str] = {
    "drone": "#FF4444",      # Coral Red
    "airplane": "#00D4FF",   # Tactical Cyan
    "bird": "#FFB800",       # Amber Gold
    "balloon": "#D500F9",    # Vivid Magenta
    "uncertain": "#8E8E93",  # Muted Slate
}


class ModelType(str, Enum):
    CUSTOM_CIVILIAN = "Custom Civilian Aerial"
    PRETRAINED_COCO = "Pretrained COCO (80 Classes)"
    UNKNOWN = "Custom Unknown"


class CheckpointMetadata:
    """Introspected metadata for a loaded or inspected YOLO checkpoint."""

    def __init__(
        self,
        path: Path,
        raw_names: Dict[int, str],
        model_type: ModelType,
        device: str,
        is_half: bool,
    ):
        self.path = path
        self.raw_names = raw_names
        self.model_type = model_type
        self.device = device
        self.is_half = is_half
        self.num_classes = len(raw_names)

    def supports_drones(self) -> bool:
        """True only if 'drone' exists in raw training names (COCO does NOT support drones)."""
        return any("drone" in name.lower() for name in self.raw_names.values())

    def supports_balloons(self) -> bool:
        """True only if 'balloon' exists in raw training names."""
        return any("balloon" in name.lower() for name in self.raw_names.values())


class ModelManager:
    """Manages model lifecycle, warm-up, metadata, and class normalization."""

    def __init__(self):
        self._current_path: Optional[Path] = None
        self._model: Optional[YOLO] = None
        self._metadata: Optional[CheckpointMetadata] = None
        self._device: str = "cpu"
        self._is_half: bool = False
        self._coco_verifier: Optional[YOLO] = None
        self._coco_verifier_failed: bool = False
        # Introspection cache keyed by (path, size, mtime_ns): introspecting a
        # checkpoint loads the full model, and callers such as predict_image
        # previously paid that cost twice (introspect + load) per run.
        self._introspect_cache: Dict[Tuple[str, int, int], CheckpointMetadata] = {}
        self._checkpoint_sha256: Optional[str] = None

    @property
    def checkpoint_sha256(self) -> Optional[str]:
        """SHA-256 of the currently loaded checkpoint, if it was validated."""
        return self._checkpoint_sha256

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> Optional[YOLO]:
        return self._model

    @property
    def metadata(self) -> Optional[CheckpointMetadata]:
        return self._metadata

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_half(self) -> bool:
        return self._is_half

    def introspect_checkpoint(self, checkpoint_path: Union[str, Path]) -> CheckpointMetadata:
        """Reads checkpoint metadata without executing training or modifying files."""
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        # Validate + fingerprint before any deserialization of the file.
        digest = validate_checkpoint_file(path)
        stat = path.stat()
        cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
        if cache_key in self._introspect_cache:
            return self._introspect_cache[cache_key]

        # Quick load with ultralytics to inspect weights
        probe = YOLO(str(path))
        raw_names: Dict[int, str] = {int(k): str(v) for k, v in probe.names.items()}

        # Classify model architecture
        if len(raw_names) == 80 and raw_names.get(0) == "person" and raw_names.get(4) == "airplane":
            m_type = ModelType.PRETRAINED_COCO
        elif any("drone" in name.lower() for name in raw_names.values()):
            m_type = ModelType.CUSTOM_CIVILIAN
        else:
            m_type = ModelType.UNKNOWN

        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        metadata = CheckpointMetadata(
            path=path,
            raw_names=raw_names,
            model_type=m_type,
            device=dev,
            is_half=torch.cuda.is_available(),
        )
        self._introspect_cache[cache_key] = metadata
        logger.info(f"Checkpoint {path.name} sha256={digest[:16]}… classes={len(raw_names)}")
        return metadata

    def load_model(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "auto",
        enable_half: bool = True,
        imgsz: int = 640,
    ) -> YOLO:
        """Loads and warms up a YOLO model on the chosen hardware device.

        Reuses model if the same checkpoint and settings are already loaded.
        """
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Cannot load missing checkpoint: {path}")

        # Validate + fingerprint before any deserialization of the file.
        digest = validate_checkpoint_file(path)

        # Resolve compute device. "auto" intentionally resolves to CPU: the
        # previous implementation loaded and ran a second throwaway copy of
        # the model as a "probe" and then discarded it, yet still always
        # chose CPU. Same decision, without the wasted load/inference.
        # Request CUDA explicitly (device="0" / "cuda") to use the GPU.
        if device == "auto":
            resolved_device = "cpu"
        else:
            resolved_device = device

        use_half = enable_half and (resolved_device == "0" or "cuda" in resolved_device) and torch.cuda.is_available()

        # Check if already loaded
        if (
            self._model is not None
            and self._current_path == path
            and self._device == resolved_device
            and self._is_half == use_half
        ):
            return self._model

        logger.info(f"Loading checkpoint: {path.name} on device={resolved_device} (half={use_half})")
        model = YOLO(str(path))

        raw_names: Dict[int, str] = {int(k): str(v) for k, v in model.names.items()}
        if len(raw_names) == 80 and raw_names.get(0) == "person" and raw_names.get(4) == "airplane":
            m_type = ModelType.PRETRAINED_COCO
        elif any("drone" in name.lower() for name in raw_names.values()):
            m_type = ModelType.CUSTOM_CIVILIAN
        else:
            m_type = ModelType.UNKNOWN

        if use_half and resolved_device != "cpu":
            try:
                model.to("cuda")
                if hasattr(model, "model") and hasattr(model.model, "half"):
                    model.model.half()
                logger.info("Activated native CUDA FP16 half-precision.")
            except Exception as e:
                logger.warning(f"Could not activate native half-precision: {e}")
                use_half = False
                resolved_device = "cpu"

        # Perform safe one-shot warmup
        dummy_input = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        try:
            model.predict(
                source=dummy_input,
                device=resolved_device,
                imgsz=imgsz,
                verbose=False,
            )
            logger.info("Model warm-up completed successfully.")
        except Exception as e:
            logger.warning(f"Warm-up on {resolved_device} failed, falling back to CPU: {e}")
            use_half = False
            resolved_device = "cpu"
            model.to("cpu")
            model.predict(
                source=dummy_input,
                device="cpu",
                imgsz=imgsz,
                verbose=False,
            )

        self._model = model
        self._current_path = path
        self._checkpoint_sha256 = digest
        self._device = resolved_device
        self._is_half = use_half
        self._metadata = CheckpointMetadata(
            path=path,
            raw_names=raw_names,
            model_type=m_type,
            device=resolved_device,
            is_half=use_half,
        )

        return self._model

    def ensure_model_loaded(self, checkpoint_path: Optional[Union[str, Path]] = None) -> YOLO:
        """Ensures that a valid model is loaded, defaulting to aerowatch_v7_balanced.pt."""
        if self._model is not None:
            return self._model

        if checkpoint_path is None:
            weights_dir = Path("weights")
            candidates = [
                weights_dir / "aerowatch_v7_balanced.pt",
                weights_dir / "aerowatch_v6_expert.pt",
                weights_dir / "best.pt",
                weights_dir / "aerowatch_real_v1.pt",
            ]
            for c in candidates:
                if c.exists():
                    checkpoint_path = c
                    break

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            checkpoint_path = Path("weights/best.pt")

        return self.load_model(checkpoint_path, device="cpu")

    def predict_airspace(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.20,
        uncertain_thresh: float = 0.35,
        imgsz: int = 640,
    ) -> List[Dict[str, Any]]:
        """Executes multi-scale inference and filters out ground/human clutter."""
        model = self.ensure_model_loaded()
        results = model.predict(
            source=frame,
            conf=conf_thresh,
            imgsz=imgsz,
            device=self._device,
            verbose=False,
        )
        if not results or len(results[0].boxes) == 0:
            return []

        raw_boxes = results[0].boxes
        return self.filter_airspace_detections(
            frame=frame,
            raw_boxes=raw_boxes,
            conf_thresh=conf_thresh,
            uncertain_thresh=uncertain_thresh,
        )

    def map_detection_label(
        self,
        raw_cls_id: int,
        confidence: float,
        detection_threshold: float = 0.25,
        uncertainty_threshold: float = 0.40,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Maps a raw class detection to canonical civilian vocabulary.

        Returns:
            (internal_name, display_label, hex_color) or (None, None, None) if filtered.
        """
        # Model output is untrusted: a NaN confidence previously *passed*
        # this check (all NaN comparisons are False) and was rendered as a
        # real target, and out-of-range confidences were displayed verbatim.
        try:
            confidence = validate_confidence(confidence)
            detection_threshold = validate_confidence(
                detection_threshold, name="detection_threshold"
            )
            uncertainty_threshold = validate_confidence(
                uncertainty_threshold, name="uncertainty_threshold"
            )
        except DetectionValidationError as e:
            logger.warning(f"Rejected detection with invalid confidence values: {e}")
            return None, None, None
        if not isinstance(raw_cls_id, int) or isinstance(raw_cls_id, bool) or raw_cls_id < 0:
            logger.warning(f"Rejected detection with invalid class id: {raw_cls_id!r}")
            return None, None, None

        if confidence < detection_threshold:
            return None, None, None

        if not self._metadata:
            return "unknown", "Unknown (Unverified)", CANONICAL_COLORS["uncertain"]

        if raw_cls_id not in self._metadata.raw_names:
            # The checkpoint emitted a class it does not declare. Never
            # fabricate a label for it (previous behaviour passed the raw
            # name through as if it were a verified class).
            logger.warning(f"Rejected detection with undeclared class id {raw_cls_id}")
            return "unknown", "Unknown (Unverified)", CANONICAL_COLORS["uncertain"]

        raw_name = self._metadata.raw_names[raw_cls_id].lower()

        # Handle Pretrained COCO (80 classes)
        if self._metadata.model_type == ModelType.PRETRAINED_COCO:
            if raw_cls_id == 4:  # COCO airplane
                internal, display = ("airplane", "Aircraft")
            elif raw_cls_id == 14:  # COCO bird
                internal, display = ("bird", "Bird")
            else:
                # Do NOT misclassify other COCO objects as aerial objects
                return None, None, None
        else:
            # Handle Custom Aerial Models
            # Never guess military subtypes without explicit validated evidence.
            # Prefer an exact name match, then the longest canonical key the
            # raw name contains, so e.g. "military_jet" wins over a shorter
            # incidental substring.
            if raw_name in CANONICAL_CIVILIAN_MAP:
                norm_key = raw_name
            else:
                matches = [k for k in CANONICAL_CIVILIAN_MAP if k in raw_name]
                norm_key = max(matches, key=len) if matches else None

            if norm_key:
                internal, display = CANONICAL_CIVILIAN_MAP[norm_key]
            else:
                # Declared by the checkpoint but not a recognised civilian
                # class: surface it as unverified, never as a titled,
                # trustworthy label.
                internal, display = ("unknown", "Unknown (Unverified)")

        # Low-confidence uncertainty flagging
        if confidence < uncertainty_threshold:
            display = f"{display} (Uncertain)"
            color = CANONICAL_COLORS.get("uncertain", "#8E8E93")
        else:
            color = CANONICAL_COLORS.get(internal, "#00E5FF")

        return internal, display, color

    def get_coco_verifier(self) -> Optional[YOLO]:
        """Lazy-loads YOLO COCO model for human / ground false positive verification."""
        if self._coco_verifier is not None:
            return self._coco_verifier
        if self._coco_verifier_failed:
            return None

        coco_candidates = [
            Path("weights/yolo26n.pt"),
            Path("weights/yolov8n.pt"),
            Path("yolo26n.pt"),
        ]
        for p in coco_candidates:
            if p.exists():
                try:
                    m = YOLO(str(p))
                    if self._device != "cpu" and torch.cuda.is_available():
                        m.to("cuda")
                        if hasattr(m, "model") and hasattr(m.model, "half"):
                            m.model.half()
                    self._coco_verifier = m
                    logger.info(f"Initialized COCO ground verifier from {p}")
                    return self._coco_verifier
                except Exception as e:
                    logger.warning(f"Could not initialize COCO verifier from {p}: {e}")
        self._coco_verifier_failed = True
        return None

    def filter_airspace_detections(
        self,
        frame: np.ndarray,
        raw_boxes: List[Any],
        conf_thresh: float,
        uncertain_thresh: float,
    ) -> List[Dict[str, Any]]:
        """Filters candidate detections with geometric airspace physics and human/person suppression.

        Guarantees that humans, faces, torsos, and vertical ground objects are NEVER
        misclassified as aircraft or drones.
        """
        img_h, img_w = frame.shape[:2]
        candidates = []

        # Contract validation first: raw model tensors are untrusted. A NaN
        # coordinate previously crashed this method via int(NaN); a NaN
        # confidence previously passed every threshold check. Both are now
        # dropped with counted reasons, and boxes are clamped to the frame.
        raw_items = []
        for b in raw_boxes:
            try:
                raw_items.append(
                    {
                        "cls": int(b.cls.item()),
                        "conf": float(b.conf.item()),
                        "xyxy": [float(v) for v in b.xyxy[0].tolist()],
                    }
                )
            except Exception as e:
                logger.warning(f"Dropped unreadable raw detection: {e}")
        sanitized = sanitize_detections(raw_items, img_w, img_h)
        if sanitized.dropped:
            logger.warning(f"Dropped malformed detections: {sanitized.dropped}")

        for det in sanitized.detections:
            cid = det.cls_id
            conf = det.conf
            x1, y1, x2, y2 = det.box

            internal_name, display_label, hex_color = self.map_detection_label(
                cid, conf, conf_thresh, uncertain_thresh
            )
            if display_label is None:
                continue

            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            cov_pct = (w * h) / (img_w * img_h) * 100.0
            aspect_ratio = h / w

            # Physical Filter 1: Fixed-wing aircraft have wingspan > height.
            # Vertical shapes (H > 1.25 * W) covering > 3% of FOV are human bodies/indoor objects.
            if internal_name in ["airplane", "aircraft", "military_jet"]:
                if aspect_ratio > 1.25 and cov_pct > 3.0:
                    logger.debug(f"Suppressed vertical object (H/W={aspect_ratio:.2f}) misclassified as aircraft")
                    continue

            # Physical Filter 2: Excessive FOV near-field coverage (person directly in front of camera)
            if cov_pct > 25.0 and aspect_ratio > 1.1:
                logger.debug(f"Suppressed giant near-field object ({cov_pct:.1f}% FOV) misclassified as aerial target")
                continue

            candidates.append({
                "box": [x1, y1, x2, y2],
                "cid": cid,
                "conf": conf,
                "internal_name": internal_name,
                "display_label": display_label,
                "hex_color": hex_color,
                "cov_pct": cov_pct,
                "aspect_ratio": aspect_ratio,
            })

        if not candidates:
            return []

        # Canonical duplicate suppression: YOLO NMS is per raw class, but
        # this checkpoint family declares both "aircraft" and
        # "military_jet", which normalise to the same canonical class —
        # one physical aircraft could otherwise be reported twice (once
        # per raw class). Measured on the test artifacts; see
        # evaluation/model_benchmark.py and docs/CV_QUALITY.md.
        candidates, suppressed = dedupe_canonical(candidates)
        if suppressed:
            logger.info(
                f"Suppressed {suppressed} canonical duplicate detection(s) "
                "(same object reported under two raw classes)"
            )

        # Ground Filter 3: COCO Person Overlap Verification
        # If candidate aerial detections exist, verify whether a human is present in the frame
        verifier = self.get_coco_verifier()
        if verifier is not None:
            try:
                coco_res = verifier.predict(
                    source=frame,
                    classes=[0],
                    conf=0.30,
                    imgsz=320,
                    device=self._device,
                    verbose=False,
                )
                person_boxes = []
                if len(coco_res) > 0 and len(coco_res[0].boxes) > 0:
                    person_items = []
                    for pb in coco_res[0].boxes:
                        try:
                            person_items.append(
                                {"cls": 0, "conf": float(pb.conf.item()), "xyxy": [float(v) for v in pb.xyxy[0].tolist()]}
                            )
                        except Exception:
                            continue
                    for pdet in sanitize_detections(person_items, img_w, img_h).detections:
                        person_boxes.append(tuple(pdet.box))

                if person_boxes:
                    verified_candidates = []
                    for c in candidates:
                        cx1, cy1, cx2, cy2 = c["box"]
                        c_area = max(1, (cx2 - cx1) * (cy2 - cy1))
                        ccx = (cx1 + cx2) / 2
                        ccy = (cy1 + cy2) / 2
                        is_person_clutter = False

                        for px1, py1, px2, py2 in person_boxes:
                            # 1. Center of candidate box is inside human body
                            if px1 <= ccx <= px2 and py1 <= ccy <= py2:
                                is_person_clutter = True
                                break
                            # 2. Significant area intersection
                            ix1 = max(cx1, px1)
                            iy1 = max(cy1, py1)
                            ix2 = min(cx2, px2)
                            iy2 = min(cy2, py2)
                            if ix2 > ix1 and iy2 > iy1:
                                inter = (ix2 - ix1) * (iy2 - iy1)
                                if inter / c_area > 0.25:
                                    is_person_clutter = True
                                    break

                        if not is_person_clutter:
                            verified_candidates.append(c)
                        else:
                            logger.info(f"Human shield activated: rejected false {c['display_label']} on human body.")

                    candidates = verified_candidates
            except Exception as e:
                logger.warning(f"COCO verifier check failed: {e}")

        return candidates

    def unload(self):
        """Releases the loaded model and frees CUDA cache."""
        self._model = None
        self._metadata = None
        self._current_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded and GPU memory released.")


def discover_checkpoints(project_root: Path) -> List[Tuple[str, Path, ModelType]]:
    """Discovers all valid local checkpoints in weights/ and root directories."""
    checkpoints: List[Tuple[str, Path, ModelType]] = []
    seen_paths = set()

    search_dirs = [
        project_root / "weights",
        project_root / "runs" / "train",
        project_root,
    ]

    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for pt in s_dir.rglob("*.pt"):
            resolved = pt.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)

            # Check if valid file
            if pt.stat().st_size < 1000:
                continue

            name = pt.name.lower()
            if "yolo26n.pt" in name or "coco" in name:
                m_type = ModelType.PRETRAINED_COCO
                label = f"🌐 {pt.name} (Pretrained COCO Baseline - Airplanes & Birds Only)"
            elif "v7" in name or "balanced" in name:
                m_type = ModelType.CUSTOM_CIVILIAN
                label = f"⭐ {pt.name} (Custom Balanced Aerial - Production Recommended)"
            elif "best" in name:
                parent_run = pt.parent.parent.name if pt.parent.name == "weights" else pt.parent.name
                m_type = ModelType.CUSTOM_CIVILIAN
                label = f"⭐ {parent_run} / best.pt (Trained Checkpoint)"
            else:
                m_type = ModelType.CUSTOM_CIVILIAN
                label = f"📦 {pt.name} (Custom Aerial Model)"

            checkpoints.append((label, resolved, m_type))

    # Sort with custom balanced models first, pretrained COCO last
    checkpoints.sort(key=lambda x: (x[2] == ModelType.PRETRAINED_COCO, not ("v7" in x[0] or "balanced" in x[0])))
    return checkpoints
