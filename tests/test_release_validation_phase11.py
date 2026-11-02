"""Phase 11: unified release-validation report semantics."""

import json

from release_validation import NOT_VERIFIED, PASS, run_validation


def test_quick_validation_report_semantics(tmp_path):
    report = run_validation(out_dir=tmp_path)
    assert report["failures"] == []
    by_name = {s["name"]: s["status"] for s in report["sections"]}
    assert by_name["preflight"] == PASS
    assert by_name["database"] == PASS
    assert by_name["app_startup"] == PASS
    # Unrequested hardware-dependent checks are NOT VERIFIED, never PASS.
    assert by_name["hardware_benchmark"] == NOT_VERIFIED
    assert by_name["camera"] == NOT_VERIFIED
    assert by_name["offline_mode"] == NOT_VERIFIED
    # Reports are written in both formats and agree.
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["overall"] == report["overall"]
    assert "NOT VERIFIED" in (tmp_path / "report.md").read_text()
