"""Phase 8: reporting/export correctness and dashboard metric accuracy."""

import json

import pytest

from history_store import HistoryStore
from reporting import (
    PREDICTION_DISCLAIMER,
    confidence_band,
    dashboard_metrics,
    detections_to_csv,
    history_to_json,
    session_report_markdown,
    session_to_json,
)


@pytest.fixture()
def store(tmp_path):
    with HistoryStore(tmp_path / "h.db") as s:
        sid = s.start_session(
            kind="image", source_label="field.jpg",
            checkpoint_name="aerowatch_v7_balanced.pt",
            checkpoint_sha256="992080afb8f38674ddbf", backend="onnx",
        )
        s.add_detections(sid, [
            {"internal_name": "drone", "display_label": "Drone",
             "conf": 0.9, "posture": "requires_verification",
             "box": [1, 2, 30, 40]},
            {"internal_name": "bird", "display_label": "Bird",
             "conf": 0.3, "posture": "uncertain", "box": [5, 6, 20, 22]},
        ], frame_width=640, frame_height=480)
        yield s, sid


def test_dashboard_metrics_empty_store(tmp_path):
    with HistoryStore(tmp_path / "empty.db") as s:
        metrics = dashboard_metrics(s)
    assert metrics["has_data"] is False
    assert metrics["totals"]["sessions"] == 0
    assert metrics["class_distribution"] == {}
    assert sum(metrics["confidence_histogram"]) == 0
    assert metrics["recent_sessions"] == []


def test_dashboard_metrics_accuracy(store):
    s, _sid = store
    metrics = dashboard_metrics(s)
    assert metrics["has_data"] is True
    assert metrics["totals"]["stored_detections"] == 2
    assert metrics["totals"]["unverified_detections"] == 2
    assert metrics["class_distribution"] == {"bird": 1, "drone": 1}
    assert metrics["recent_sessions"][0]["backend"] == "onnx"


def test_csv_export(store):
    s, sid = store
    csv_text = detections_to_csv(s.session_detections(sid))
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("detection_id,session_id,created_at")
    assert len(lines) == 3
    assert "Drone" in lines[1] and "0.9000" in lines[1]
    assert "unverified" in lines[2]


def test_json_exports_carry_disclaimer(store):
    s, sid = store
    session = s.get_session(sid)
    one = json.loads(session_to_json(session, s.session_detections(sid)))
    assert one["session"]["session_id"] == sid
    assert len(one["detections"]) == 2
    assert "unverified" in one["disclaimer"]
    whole = json.loads(history_to_json(s))
    assert len(whole["sessions"]) == 1
    assert len(whole["detections"]) == 2


def test_session_report_markdown(store):
    s, sid = store
    session = s.get_session(sid)
    md = session_report_markdown(session, s.session_detections(sid))
    assert "# AeroWatch AI — Session Report" in md
    assert "aerowatch_v7_balanced.pt" in md
    assert "992080afb8f3" in md
    assert "onnx" in md
    assert PREDICTION_DISCLAIMER in md
    assert "confirmed hostile" not in md.lower().replace(
        "not a confirmed", "")


def test_confidence_band_language():
    low = confidence_band(0.3, 0.5)
    assert "Uncertain" in low and "below the uncertainty threshold" in low
    high = confidence_band(0.9, 0.5)
    assert "not a certainty" in high
    # The band text must never convert a score into hostility language.
    for text in (low, high):
        assert "hostile" not in text.lower()
        assert "probability" not in text.lower() or "not" in text.lower()
