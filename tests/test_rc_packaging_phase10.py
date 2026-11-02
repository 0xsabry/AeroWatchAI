"""Phase 10: preflight, offline guard, backup/restore, diagnostics."""

import socket
import sqlite3
from pathlib import Path

import pytest

import offline_guard
from history_store import HistoryStore
from preflight import EXPECTED_V7_SHA256, run_preflight


# -- preflight ---------------------------------------------------------

def test_preflight_passes_in_repo_environment():
    report = run_preflight()
    assert report["overall"] == "ok"
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["checkpoint_v7"]["status"] == "ok"
    assert EXPECTED_V7_SHA256.startswith("992080af")


# -- offline guard -------------------------------------------------------

@pytest.fixture()
def guard():
    offline_guard.enable_offline_mode()
    yield
    offline_guard.disable_offline_mode()


def test_offline_guard_blocks_external(guard):
    with pytest.raises(OSError, match="offline mode"):
        socket.create_connection(("8.8.8.8", 53), timeout=2)


def test_offline_guard_allows_loopback(guard):
    import threading

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen(1)
    threading.Thread(target=lambda: server.accept(), daemon=True).start()
    conn = socket.create_connection(("127.0.0.1", port), timeout=2)
    conn.close()
    server.close()


def test_offline_not_enabled_by_default():
    assert offline_guard.is_enabled() is False
    assert offline_guard.offline_requested() is False


# -- backup / restore ----------------------------------------------------

def _seed(store):
    sid = store.start_session(kind="image", source_label="a.jpg")
    store.add_detections(sid, [{"internal_name": "drone",
                                "display_label": "Drone", "conf": 0.9,
                                "posture": "requires_verification",
                                "box": [0, 0, 5, 5]}])
    return sid


def test_backup_restore_roundtrip(tmp_path):
    store = HistoryStore(tmp_path / "live.db")
    _seed(store)
    backup = store.backup_to(tmp_path / "backup.db")
    assert backup.exists()
    # Corrupt the live DB by deleting its data, then restore.
    store.clear_all()
    assert store.totals()["sessions"] == 0
    with pytest.raises(ValueError, match="confirm=True"):
        store.restore_from(backup)
    result = store.restore_from(backup, confirm=True)
    assert store.totals()["sessions"] == 1
    assert store.query_detections()[0].canonical_class == "drone"
    # The pre-restore safety copy exists and holds the (empty) state.
    assert (tmp_path / Path(result["protected_copy"]).name).exists()
    store.close()


def test_restore_rejects_corrupt_and_wrong_schema(tmp_path):
    store = HistoryStore(tmp_path / "live.db")
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"this is not sqlite at all" * 10)
    with pytest.raises(Exception):
        store.restore_from(bad, confirm=True)
    wrong = tmp_path / "wrong.db"
    conn = sqlite3.connect(str(wrong))
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="schema mismatch"):
        store.restore_from(wrong, confirm=True)
    with pytest.raises(ValueError, match="differ"):
        store.backup_to(tmp_path / "live.db")
    store.close()


def test_verify_integrity(tmp_path):
    store = HistoryStore(tmp_path / "live.db")
    _seed(store)
    info = store.verify_integrity()
    assert info["ok"] and info["sessions"] == 1 and info["detections"] == 1
    store.close()


# -- diagnostics & camera tool (simulated paths only) --------------------

def test_hardware_diagnostics_collects_real_facts():
    from evaluation.hardware_diagnostics import collect

    report = collect()
    assert report["checkpoint_verified"] is True
    assert report["recommended_config"]["imgsz"] == 640
    assert report["cpu_cores"] >= 1
    # This dev environment has no GPU: the tool must say so, not invent.
    assert report["cuda_available"] is False
    assert report["gpu_name"] is None


def test_camera_check_simulated_mode_is_labelled():
    from evaluation.camera_check import run_simulated

    report = run_simulated()
    assert report["hardware_verified"] is False
    assert "SIMULATED" in report["mode"]
    assert report["history_roundtrip_ok"] is True
    assert report["frames_processed"] == 15
