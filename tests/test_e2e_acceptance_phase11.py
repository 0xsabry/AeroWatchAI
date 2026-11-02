"""Phase 11: end-to-end release acceptance (headless, real components).

Chains the actual production pieces — no UI, no mocks:
image → inference (PyTorch AND ONNX) → safety filter → posture →
history → dashboard metrics → CSV export → restart persistence;
video → validation → processing → aggregate-only history;
offline guard active during inference; invalid-input handling.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from ai_safety import DetectionValidationError, assess_posture, validate_upload
from history_store import HistoryStore
from model_manager import ModelManager
from reporting import dashboard_metrics, detections_to_csv

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "videos" / "test_soldier_drone_photo.jpg"


@pytest.fixture(scope="module")
def mm():
    manager = ModelManager()
    manager.ensure_model_loaded()
    return manager


def _detect(mm, image_path):
    frame = cv2.imread(str(image_path))
    results = mm.model.predict(source=frame, device=mm.device, conf=0.25,
                               imgsz=640, iou=0.45, verbose=False)
    return mm.filter_airspace_detections(
        frame=frame, raw_boxes=results[0].boxes,
        conf_thresh=0.25, uncertain_thresh=0.50)


def test_image_chain_torch_to_export_and_restart(mm, tmp_path):
    dets = _detect(mm, SAMPLE)
    assert dets, "sample photo must produce detections"
    postures = [assess_posture(d["internal_name"], d["conf"],
                               uncertainty_threshold=0.50) for d in dets]
    assert all(p.confirmed_hostile is False for p in postures)
    assert all(p.auto_engage_allowed is False for p in postures)

    db = tmp_path / "e2e.db"
    store = HistoryStore(db)
    sid = store.start_session(
        kind="image", source_label=SAMPLE.name,
        checkpoint_name="aerowatch_v7_balanced.pt",
        checkpoint_sha256=mm.checkpoint_sha256, backend=mm.backend)
    rows = [{**d, "posture": p.posture.value}
            for d, p in zip(dets, postures)]
    assert store.add_detections(sid, rows) == len(dets)
    metrics = dashboard_metrics(store)
    assert metrics["totals"]["stored_detections"] == len(dets)
    csv_text = detections_to_csv(store.session_detections(sid))
    assert "Drone" in csv_text
    store.close()

    reopened = HistoryStore(db)  # simulated application restart
    assert reopened.totals()["stored_detections"] == len(dets)
    assert reopened.verify_integrity()["ok"]
    reopened.close()


def test_image_chain_onnx_backend(mm):
    onnx_path = ROOT / "weights" / "aerowatch_v7_balanced.onnx"
    if not onnx_path.exists():
        pytest.skip("ONNX artifact not present")
    mm.load_onnx_model(onnx_path)
    assert mm.backend == "onnxruntime"  # ModelManager's backend label
    dets = _detect(mm, SAMPLE)
    assert dets, "ONNX backend must detect on the sample photo"


def test_video_chain_aggregates_only(mm, tmp_path):
    from video_processor import VideoProcessor

    clip = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (320, 240))
    for i in range(12):
        writer.write(np.full((240, 320, 3), i * 8, dtype=np.uint8))
    writer.release()
    validate_upload(clip.name, clip.stat().st_size, kind="video")

    processor = VideoProcessor(mm)
    metrics = processor.process_video(clip, max_frames=12)
    assert metrics.frames_processed == 12

    store = HistoryStore(tmp_path / "vid.db")
    sid = store.start_session(kind="video", source_label=clip.name)
    store.record_class_counts(sid, dict(metrics.detections_by_class))
    session = store.get_session(sid)
    assert session.detection_count == sum(
        metrics.detections_by_class.values())
    # Aggregate-only: no individual rows may exist for a video session.
    assert store.session_detections(sid) == []
    store.close()


def test_offline_guard_during_inference(mm):
    import offline_guard

    offline_guard.enable_offline_mode()
    try:
        dets = _detect(mm, SAMPLE)
        assert dets  # local inference unaffected by the guard
    finally:
        offline_guard.disable_offline_mode()


def test_invalid_inputs_and_empty_state(tmp_path):
    with pytest.raises(DetectionValidationError):
        validate_upload("evil.exe", 100, kind="image")
    with pytest.raises(DetectionValidationError):
        validate_upload("big.mp4", 10 * 1024 * 1024 * 1024, kind="video")
    store = HistoryStore(tmp_path / "empty.db")
    metrics = dashboard_metrics(store)
    assert metrics["has_data"] is False
    assert metrics["totals"]["sessions"] == 0
    store.close()
