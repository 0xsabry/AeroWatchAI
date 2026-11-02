"""Phase 9: app-wiring guards for stabilization changes."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
REPORTING = (ROOT / "reporting.py").read_text(encoding="utf-8")


def test_camera_session_lifecycle_wired():
    assert "from camera_session import CameraSessionTracker" in APP
    assert "_finalize_camera_session" in APP
    assert 'kind="camera"' in APP
    assert "cam_tracker" in APP
    # Interrupted sessions are recorded honestly.
    assert "ended_cleanly" in APP


def test_mobile_sidebar_not_forced_open():
    assert 'initial_sidebar_state="auto"' in APP
    assert 'initial_sidebar_state="expanded"' not in APP


def test_latency_card_uses_real_device_not_hardcoded_gpu():
    assert "Forward pass on RTX 3050" not in APP
    assert 'getattr(mm, "device", "cpu")' in APP


def test_csv_formula_guard_present():
    assert "_csv_safe" in REPORTING


def test_history_date_filters_wired():
    assert "From date (optional)" in APP
    assert "date_from=" in APP and "date_to=" in APP
    assert '"Max results"' in APP
