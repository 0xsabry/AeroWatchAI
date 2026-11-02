"""Phase 4 ONNX parity gates.

The ONNX backend may only be considered for default use if these hold.
Tests skip cleanly when onnxruntime or the exported model is absent
(e.g. minimal CI), so the suite never fails for a missing *optional*
artifact — but when both are present, parity is enforced.

Measured basis (docs/RUNTIME_PHASE4.md): raw output tensors differ by
at most ~2e-3 (FP32 kernel-order drift), so disagreements are confined
to borderline candidates near the confidence threshold. The gates
therefore require: every *confident* (>= 0.50) PyTorch detection to be
matched by ONNX (same canonical class, IoU >= 0.5), and overall
agreement >= 90% at the operating point.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "weights" / "aerowatch_v7_balanced.onnx"
PT_PATH = ROOT / "weights" / "aerowatch_v7_balanced.pt"

ort_missing = False
try:
    import onnxruntime  # noqa: F401
except Exception:
    ort_missing = True

pytestmark = pytest.mark.skipif(
    ort_missing or not ONNX_PATH.exists(),
    reason="onnxruntime or the ONNX export is not available",
)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


@pytest.fixture(scope="module")
def both_backends():
    from model_manager import ModelManager

    mm_pt = ModelManager()
    mm_pt.load_model(PT_PATH, device="cpu")
    mm_ox = ModelManager()
    mm_ox.load_onnx_model(ONNX_PATH)
    return mm_pt, mm_ox


def _filter_dets(mm, img):
    res = mm.model.predict(img, conf=0.25, iou=0.45, imgsz=640, verbose=False)[0]
    return mm.filter_airspace_detections(img, res.boxes, 0.25, 0.50)


def test_onnx_backend_identity_and_names(both_backends):
    mm_pt, mm_ox = both_backends
    assert mm_ox.backend == "onnxruntime"
    assert mm_pt.backend == "pytorch"
    # Canonical class vocabulary preserved through the export.
    assert mm_ox.metadata.raw_names == mm_pt.metadata.raw_names
    assert mm_ox.checkpoint_sha256 is not None and len(mm_ox.checkpoint_sha256) == 64


def test_onnx_parity_through_safety_filter(both_backends):
    from PIL import Image

    mm_pt, mm_ox = both_backends
    img_dir = ROOT / "tests" / "artifacts" / "images"
    sample = sorted(img_dir.glob("*.jpg"))[::6]  # deterministic spread of 5
    assert len(sample) >= 4
    total_pt = matched = 0
    confident_unmatched = []
    for path in sample:
        img = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]
        d_pt = _filter_dets(mm_pt, img)
        d_ox = _filter_dets(mm_ox, img)
        used: set[int] = set()
        for a in d_pt:
            total_pt += 1
            best, bj = 0.0, -1
            for j, b in enumerate(d_ox):
                if b["internal_name"] == a["internal_name"] and j not in used:
                    v = _iou(a["box"], b["box"])
                    if v > best:
                        best, bj = v, j
            if bj >= 0 and best >= 0.5:
                matched += 1
                used.add(bj)
            elif a["conf"] >= 0.50:
                confident_unmatched.append((path.name, a["display_label"], a["conf"]))
    assert not confident_unmatched, f"confident torch detections lost by ONNX: {confident_unmatched}"
    assert total_pt == 0 or matched / total_pt >= 0.9


def test_onnx_detections_pass_safety_contract(both_backends):
    """ONNX detections are subject to the same ai_safety posture rules."""
    from ai_safety import ThreatPosture, assess_posture

    _mm_pt, mm_ox = both_backends
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    dets = _filter_dets(mm_ox, img)  # empty scene: must not raise, may be []
    for d in dets:
        posture = assess_posture(d["internal_name"], d["conf"], uncertainty_threshold=0.50)
        assert posture.posture in set(ThreatPosture)
        assert posture.confirmed_hostile is False
        assert posture.auto_engage_allowed is False


def test_missing_onnxruntime_error_is_clear(monkeypatch):
    """Without onnxruntime, the backend fails with guidance, not a stack trace."""
    import onnx_backend

    monkeypatch.setattr(onnx_backend, "ORT_AVAILABLE", False)
    monkeypatch.setattr(onnx_backend, "ORT_IMPORT_ERROR", ImportError("simulated"))
    with pytest.raises(RuntimeError, match="onnxruntime is not installed"):
        onnx_backend.OnnxYoloWrapper(ONNX_PATH)
