"""AeroWatch AI — ONNX Runtime inference backend (Phase 4).

An alternative to the PyTorch/Ultralytics backend that:

- Loads ``.onnx`` exports with ONNX Runtime — **no pickle deserialization**,
  removing the code-execution-on-load surface of ``.pt`` checkpoints.
- Exposes the same ``predict()`` shape the rest of the codebase uses
  (results with ``.boxes`` whose items expose ``cls.item()``,
  ``conf.item()`` and ``xyxy[0].tolist()``), so detections flow through
  the *identical* ``ai_safety`` contract, canonical mapping, geometric
  filters and deduplication in ``ModelManager.filter_airspace_detections``.
  This backend cannot bypass ``ai_safety.py``: it only produces raw
  candidate boxes, exactly like the PyTorch backend.
- Implements YOLO letterbox preprocessing and per-class NMS in numpy.

The PyTorch backend remains the default. This backend is opt-in via
``ModelManager.load_onnx_model()`` and is gated by the parity tests in
``tests/test_onnx_parity.py`` (see docs/RUNTIME_PHASE4.md).

If ``onnxruntime`` is not installed, importing this module still works;
constructing a session raises a clear RuntimeError with install
instructions instead of a raw ImportError deep in the stack.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

logger = logging.getLogger("AeroWatch.OnnxBackend")

try:  # graceful degradation when the optional dependency is missing
    import onnxruntime as ort

    ORT_AVAILABLE = True
    ORT_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # pragma: no cover - exercised only without ORT
    ort = None
    ORT_AVAILABLE = False
    ORT_IMPORT_ERROR = e


class _Scalar:
    """Mimics a 0-d tensor for the .cls/.conf accessors the pipeline uses."""

    def __init__(self, value: float):
        self._value = float(value)

    def item(self) -> float:
        return self._value


class _XYXY:
    """Mimics the xyxy tensor: xyxy[0].tolist() -> [x1, y1, x2, y2]."""

    def __init__(self, box: list[float]):
        self._box = [float(v) for v in box]

    def __getitem__(self, idx: int):
        if idx != 0:
            raise IndexError(idx)
        return SimpleNamespace(tolist=lambda: list(self._box))


class OnnxBox:
    def __init__(self, cls_id: int, conf: float, box: list[float]):
        self.cls = _Scalar(cls_id)
        self.conf = _Scalar(conf)
        self.xyxy = _XYXY(box)
        self.id = None


def letterbox(img: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Ultralytics-style letterbox: scale to fit imgsz x imgsz, pad with 114."""
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    import cv2

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, (left, top)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS over xyxy boxes; returns kept indices (score order)."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(1e-9, areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thr]
    return keep


class OnnxYoloWrapper:
    """Drop-in predict() surface over an ONNX Runtime session."""

    def __init__(self, onnx_path: str | Path, device: str = "cpu"):
        if not ORT_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is not installed, so the ONNX backend is "
                f"unavailable (import error: {ORT_IMPORT_ERROR}). "
                "Install it with: pip install onnxruntime "
                "(or onnxruntime-gpu for CUDA). The PyTorch backend is "
                "unaffected."
            )
        self.path = Path(onnx_path)
        providers = ["CPUExecutionProvider"]
        self.device = "cpu"
        if device not in ("cpu", "auto"):
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.device = "cuda"
            else:
                logger.warning(
                    "CUDA requested for ONNX backend but CUDAExecutionProvider "
                    "is unavailable; falling back to CPU."
                )
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        raw_names = meta.get("names")
        if raw_names:
            parsed = ast.literal_eval(raw_names)
            self.names = {int(k): str(v) for k, v in parsed.items()}
        else:  # pragma: no cover - exports from this project always carry names
            self.names = {}
        self.nc = len(self.names)

    # -- inference ------------------------------------------------------
    def predict(
        self,
        source: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
        verbose: bool = False,
        classes: Optional[list[int]] = None,
        **_kwargs,
    ):
        if not isinstance(source, np.ndarray):
            raise TypeError(
                "ONNX backend predict() expects a numpy BGR image array; "
                f"got {type(source)}"
            )
        img = source
        canvas, scale, (pad_x, pad_y) = letterbox(img, imgsz)
        rgb = canvas[:, :, ::-1]  # BGR -> RGB
        tensor = (
            np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
            / 255.0
        )
        out = self.session.run(None, {self.input_name: tensor})[0]
        # YOLO export output: [1, 4 + nc, N] (xywh centre format + class scores)
        preds = out[0].T  # [N, 4 + nc]
        if preds.shape[1] < 5:
            return [SimpleNamespace(boxes=[])]
        scores_all = preds[:, 4:]
        cls_ids = scores_all.argmax(axis=1)
        confs = scores_all.max(axis=1)
        mask = confs >= conf
        if classes is not None:
            mask &= np.isin(cls_ids, np.asarray(classes))
        if not mask.any():
            return [SimpleNamespace(boxes=[])]
        preds, cls_ids, confs = preds[mask], cls_ids[mask], confs[mask]

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        # undo letterbox
        x1 = (x1 - pad_x) / scale
        x2 = (x2 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        y2 = (y2 - pad_y) / scale
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        kept: list[int] = []
        for cls in np.unique(cls_ids):
            idx = np.where(cls_ids == cls)[0]
            for k in _nms(boxes_xyxy[idx], confs[idx], iou):
                kept.append(int(idx[k]))
        kept.sort(key=lambda i: -confs[i])
        boxes = [
            OnnxBox(int(cls_ids[i]), float(confs[i]), boxes_xyxy[i].tolist())
            for i in kept
        ]
        return [SimpleNamespace(boxes=boxes)]

    def track(self, *_args, **_kwargs):  # pragma: no cover
        raise NotImplementedError(
            "The ONNX backend does not implement track(); use the PyTorch "
            "backend for tracking workflows (evaluate_videos.py)."
        )
