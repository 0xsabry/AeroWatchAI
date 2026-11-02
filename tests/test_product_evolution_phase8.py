"""Phase 8: app-integration guards for the product-evolution features.

Behaviour of the store/reporting layer is tested in
test_history_store.py and test_reporting.py; these tests pin the
wiring inside app.py so the features cannot silently disappear.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")


def test_new_pages_registered_in_navigation():
    for page in ('"Dashboard"', '"Detection History"',
                 '"Analytics & Reports"'):
        assert page in APP, f"nav missing {page}"
    # Dashboard is the landing page.
    assert 'st.session_state.get("nav_module", "Dashboard")' in APP


def test_history_recording_hooks_present():
    assert "get_history_store" in APP
    assert "_record_session_once" in APP
    # Image analyses and video runs are both recorded.
    assert 'kind="image"' in APP
    assert 'kind="video"' in APP
    # Recording is fingerprint-deduped against Streamlit reruns.
    assert "content_fingerprint" in APP
    assert "last_history_fingerprint" in APP


def test_dashboard_uses_real_history_not_fabricated_stats():
    assert "reporting.dashboard_metrics" in APP
    assert "No history yet" in APP  # honest empty state


def test_video_defaults_and_validation():
    # 640 (trained resolution) is the default inference size option.
    assert "640 (Trained resolution — measured best for this checkpoint)" in APP
    # Uploaded videos are size/type validated before processing.
    assert 'validate_upload(v_file.name, v_file.size, kind="video")' in APP


def test_explainability_panel_present():
    assert "Prediction details" in APP
    assert "reporting.confidence_band" in APP


def test_history_page_has_deletion_and_retention_controls():
    assert "Delete this session" in APP
    assert "Delete entire history" in APP
    assert "Retention settings" in APP
    assert "aerowatch_detections_export.csv" in APP
