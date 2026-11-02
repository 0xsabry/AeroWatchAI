"""Phase 9: camera session lifecycle, tested with SIMULATED detections.

No camera hardware exists in the test environment; these tests verify
the lifecycle/aggregation logic only. Hardware verification status is
reported separately in docs/RELEASE_CANDIDATE_PHASE9.md.
"""

from camera_session import CameraSessionTracker
from history_store import HistoryStore


def _dets(*classes):
    return [{"internal_name": c, "posture": "requires_verification"
             if c == "drone" else "non_hostile"} for c in classes]


def test_full_lifecycle_aggregates():
    t = CameraSessionTracker("camera 0")
    t.start(now=1000.0)
    assert t.is_active
    t.observe(_dets("drone", "bird"))
    t.observe(_dets("drone"))
    t.observe([])  # empty frame still counts as processed
    summary = t.stop(now=1065.0)
    assert summary["duration_sec"] == 65.0
    assert summary["frames_processed"] == 3
    assert summary["class_counts"] == {"bird": 1, "drone": 2}
    assert summary["posture_counts"] == {"non_hostile": 1,
                                         "requires_verification": 2}
    assert summary["total_detections"] == 3
    assert summary["ended_cleanly"] is True
    assert "not object tracking" in summary["counting_note"]


def test_interrupted_session_is_honest():
    t = CameraSessionTracker("camera 1")
    t.start(now=0.0)
    t.observe(_dets("aircraft"))
    summary = t.interrupt(now=5.0)
    assert summary["ended_cleanly"] is False
    assert summary["class_counts"] == {"aircraft": 1}


def test_observe_after_stop_is_ignored():
    t = CameraSessionTracker()
    t.start(now=0.0)
    t.stop(now=1.0)
    t.observe(_dets("drone"))
    assert t.summary(now=2.0)["total_detections"] == 0
    assert t.frames_processed == 0


def test_malformed_detections_skipped():
    t = CameraSessionTracker()
    t.start(now=0.0)
    t.observe([{"no_class": True}, "not-a-dict", {"internal_name": "bird"}])
    summary = t.stop(now=1.0)
    # {} without a class key counts as 'unknown'; the string is skipped.
    assert summary["class_counts"] == {"bird": 1, "unknown": 1}


def test_summary_persists_to_history_as_camera_session(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    t = CameraSessionTracker("camera 0")
    t.start(now=0.0)
    t.observe(_dets("drone", "drone", "bird"))
    summary = t.stop(now=30.0)
    sid = store.start_session(kind="camera", source_label="camera 0")
    store.record_class_counts(sid, summary["class_counts"])
    session = store.get_session(sid)
    assert session.kind == "camera"
    assert session.detection_count == 3
    assert session.class_counts == {"bird": 1, "drone": 2}
    store.close()
