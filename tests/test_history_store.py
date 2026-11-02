"""Phase 8: detection-history store correctness, isolation, retention."""

from datetime import datetime, timedelta, timezone

import pytest

from history_store import HistoryStore


@pytest.fixture()
def store(tmp_path):
    with HistoryStore(tmp_path / "h.db") as s:
        yield s


def _det(cls="Drone", conf=0.9, posture="requires_verification"):
    return {
        "internal_name": cls.lower(),
        "canonical_class": cls,
        "display_label": cls,
        "conf": conf,
        "posture": posture,
        "box": [10, 20, 110, 120],
    }


def _session(store, **kw):
    return store.start_session(
        kind=kw.get("kind", "image"),
        source_label=kw.get("source_label", "photo.jpg"),
        checkpoint_name="aerowatch_v7_balanced.pt",
        checkpoint_sha256="992080afb8f38674ddbf",
        backend="pytorch",
        started_at=kw.get("started_at"),
    )


def test_session_and_detection_roundtrip(store):
    sid = _session(store)
    assert store.add_detections(sid, [_det(), _det("Bird", 0.4, "uncertain")],
                                frame_width=640, frame_height=480) == 2
    session = store.get_session(sid)
    assert session.detection_count == 2
    assert session.class_counts == {"bird": 1, "drone": 1}
    dets = store.session_detections(sid)
    assert [d.canonical_class for d in dets] == ["drone", "bird"]
    assert dets[0].verification_status == "unverified"
    assert dets[0].box_xyxy == [10.0, 20.0, 110.0, 120.0]
    assert session.checkpoint_sha256_prefix == "992080afb8f3"


def test_source_label_stores_basename_only(store):
    sid = store.start_session(kind="image", source_label="/home/user/secret/photo.jpg")
    assert store.get_session(sid).source_label == "photo.jpg"


def test_malformed_detections_skipped_not_fabricated(store):
    sid = _session(store)
    bad = [
        {"internal_name": "drone", "conf": float("nan"), "box": [0, 0, 1, 1]},
        {"internal_name": "drone", "conf": 1.7, "box": [0, 0, 1, 1]},
        {"internal_name": "drone", "conf": "high", "box": [0, 0, 1, 1]},
        {"internal_name": "drone"},
    ]
    assert store.add_detections(sid, bad) == 0
    assert store.get_session(sid).detection_count == 0


def test_query_filters(store):
    sid = _session(store)
    store.add_detections(sid, [_det("Drone", 0.9), _det("Drone", 0.3),
                               _det("Aircraft", 0.8)])
    assert len(store.query_detections(classes=["Drone"])) == 2
    assert len(store.query_detections(min_confidence=0.5)) == 2
    assert len(store.query_detections(max_confidence=0.5)) == 1
    assert len(store.query_detections(search="airc")) == 1
    assert len(store.query_detections(search="PHOTO")) == 3  # source label
    assert len(store.query_detections(session_id=sid)) == 3
    assert len(store.query_detections(date_from="2999-01-01")) == 0


def test_store_isolation(tmp_path):
    a = HistoryStore(tmp_path / "a.db")
    b = HistoryStore(tmp_path / "b.db")
    sid = _session(a)
    a.add_detections(sid, [_det()])
    assert a.totals()["stored_detections"] == 1
    assert b.totals()["stored_detections"] == 0
    assert b.list_sessions() == []
    a.close()
    b.close()


def test_delete_and_clear(store):
    sid = _session(store)
    store.add_detections(sid, [_det()])
    assert store.delete_session(sid) is True
    assert store.get_session(sid) is None
    assert store.totals()["stored_detections"] == 0
    sid2 = _session(store)
    store.add_detections(sid2, [_det()])
    store.clear_all()
    assert store.totals()["sessions"] == 0


def test_retention_prune(store):
    old = datetime.now(timezone.utc) - timedelta(days=90)
    sid_old = _session(store, started_at=old)
    store.add_detections(sid_old, [_det()])
    sid_new = _session(store)
    store.add_detections(sid_new, [_det()])
    assert store.prune(retention_days=30) == 1
    assert store.get_session(sid_old) is None
    assert store.get_session(sid_new) is not None


def test_aggregate_class_counts_for_video_sessions(store):
    sid = store.start_session(kind="video", source_label="clip.mp4")
    store.record_class_counts(sid, {"Drone": 12, "Bird": 3, "Zero": 0})
    session = store.get_session(sid)
    assert session.detection_count == 15
    assert session.class_counts == {"Bird": 3, "Drone": 12}
    # Aggregates never fabricate individual detection rows.
    assert store.session_detections(sid) == []
    assert store.totals()["stored_detections"] == 0
    assert store.totals()["total_detections_including_aggregates"] == 15


def test_verification_status_is_explicit(store):
    sid = _session(store)
    store.add_detections(sid, [_det()])
    det = store.session_detections(sid)[0]
    assert det.verification_status == "unverified"
    store.set_verification(det.detection_id, "verified")
    assert store.session_detections(sid)[0].verification_status == "verified"
    with pytest.raises(ValueError):
        store.set_verification(det.detection_id, "confirmed_hostile")


def test_dashboard_aggregates(store):
    sid = _session(store)
    store.add_detections(sid, [_det("Drone", 0.95), _det("Drone", 0.15),
                               _det("Bird", 0.55)])
    assert store.class_distribution() == {"bird": 1, "drone": 2}
    hist = store.confidence_histogram(bins=10)
    assert hist[9] == 1 and hist[1] == 1 and hist[5] == 1
    assert sum(hist) == 3
    by_day = store.detections_by_day()
    assert sum(by_day.values()) == 3
    totals = store.totals()
    assert totals["unverified_detections"] == 3


def test_unknown_session_raises(store):
    with pytest.raises(KeyError):
        store.add_detections("nope", [_det()])
