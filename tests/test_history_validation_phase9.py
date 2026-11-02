"""Phase 9: history/analytics validation and export hardening.

- Date and combined filters, ordering, pagination limits.
- Dashboard aggregates cross-checked against direct SQL.
- No double counting between stored rows and video aggregates.
- CSV formula-injection neutralisation; Unicode/Arabic round-trips;
  empty and large exports.
"""

import csv
import io
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from history_store import HistoryStore
from reporting import (
    dashboard_metrics,
    detections_to_csv,
    history_to_json,
    session_report_markdown,
)

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def store(tmp_path):
    with HistoryStore(tmp_path / "h.db") as s:
        yield s


def _add(store, *, day_offset=0, cls="drone", conf=0.9, label=None,
         source="صورة جوية.jpg"):
    sid = store.start_session(
        kind="image", source_label=source,
        started_at=T0 + timedelta(days=day_offset))
    store.add_detections(sid, [{
        "internal_name": cls, "display_label": label or cls.title(),
        "conf": conf, "posture": "requires_verification",
        "box": [0, 0, 10, 10],
    }], created_at=T0 + timedelta(days=day_offset))
    return sid


def test_date_filters(store):
    _add(store, day_offset=0)
    _add(store, day_offset=5)
    _add(store, day_offset=10)
    assert len(store.query_detections(
        date_from="2026-11-03T00:00:00+00:00",
        date_to="2026-11-08T23:59:59+00:00")) == 1
    assert len(store.query_detections(
        date_from="2026-11-01T00:00:00+00:00")) == 3
    assert len(store.query_detections(
        date_to="2026-11-01T23:59:59+00:00")) == 1


def test_combined_filters_and_ordering(store):
    _add(store, cls="drone", conf=0.9, day_offset=0)
    _add(store, cls="drone", conf=0.4, day_offset=1)
    _add(store, cls="bird", conf=0.95, day_offset=2)
    rows = store.query_detections(classes=["DRONE"], min_confidence=0.5)
    assert len(rows) == 1 and rows[0].confidence == 0.9
    rows = store.query_detections()
    assert [r.confidence for r in rows] == [0.95, 0.4, 0.9]  # newest first
    assert len(store.query_detections(limit=2)) == 2  # pagination cap


def test_dashboard_matches_direct_sql(store, tmp_path):
    _add(store, cls="drone", conf=0.9)
    _add(store, cls="bird", conf=0.3)
    vid = store.start_session(kind="video", source_label="clip.mp4")
    store.record_class_counts(vid, {"Drone": 40, "Bird": 10})
    metrics = dashboard_metrics(store)
    raw = sqlite3.connect(str(tmp_path / "h.db"))
    sql_sessions = raw.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    sql_dets = raw.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    sql_total = raw.execute(
        "SELECT SUM(detection_count) FROM sessions").fetchone()[0]
    raw.close()
    assert metrics["totals"]["sessions"] == sql_sessions == 3
    assert metrics["totals"]["stored_detections"] == sql_dets == 2
    # Aggregates are counted once: 2 stored + 50 aggregate = 52 total,
    # and stored rows are NOT also inside the aggregate figure per-row.
    assert metrics["totals"]["total_detections_including_aggregates"] == 52
    assert metrics["class_distribution"] == {"bird": 1, "drone": 1}


def test_csv_formula_injection_neutralised(store):
    sid = store.start_session(kind="image", source_label="x.jpg")
    store.add_detections(sid, [{
        "internal_name": "drone",
        "display_label": "=2+5+cmd|' /C calc'!A0",
        "conf": 0.9, "posture": "@SUM(1+1)", "box": [0, 0, 1, 1],
    }])
    text = detections_to_csv(store.session_detections(sid))
    rows = list(csv.reader(io.StringIO(text)))
    label_cell = rows[1][4]
    posture_cell = rows[1][6]
    assert label_cell.startswith("'=")
    assert posture_cell.startswith("'@")
    assert not label_cell.startswith("=")


def test_unicode_arabic_exports_roundtrip(store):
    sid = _add(store, label="طائرة مسيرة")
    dets = store.session_detections(sid)
    csv_text = detections_to_csv(dets)
    assert "طائرة مسيرة" in csv_text
    payload = json.loads(history_to_json(store))
    assert payload["sessions"][0]["source_label"] == "صورة جوية.jpg"
    md = session_report_markdown(store.get_session(sid), dets)
    assert "Session Report" in md
    assert "unverified" in md


def test_empty_exports_are_well_formed(store):
    assert detections_to_csv([]).count("\n") == 1  # header only
    payload = json.loads(history_to_json(store))
    assert payload["sessions"] == [] and payload["detections"] == []


def test_large_export_row_count(store):
    sid = store.start_session(kind="image", source_label="bulk.jpg")
    batch = [{"internal_name": "drone", "display_label": "Drone",
              "conf": 0.5, "posture": "uncertain", "box": [0, 0, 5, 5]}
             ] * 1500
    assert store.add_detections(sid, batch) == 1500
    text = detections_to_csv(store.query_detections(limit=5000))
    assert len(text.strip().splitlines()) == 1501
