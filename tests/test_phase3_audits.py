"""Phase 3 audit tests: dataset provenance, license inventory, product claims.

These tests pin the *evidence state* of the repository: the dataset
audit's findings on the shipped artifacts, the completeness of the
license inventory, and the absence of guaranteed-outcome claims in
user-facing code. They are regression guards for the audits in
docs/DATASET_PROVENANCE.md, docs/LICENSE_AUDIT.md and
docs/PRODUCT_READINESS.md — not legal or scientific conclusions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Dataset audit utility
# ---------------------------------------------------------------------------

def test_phash_identical_images_have_zero_distance(tmp_path):
    from evaluation.dataset_audit import hamming, phash64

    img = Image.new("RGB", (64, 64), (120, 160, 200))
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    img.save(a)
    img.save(b)
    assert hamming(phash64(a), phash64(b)) == 0


def test_scan_detects_exact_duplicates(tmp_path):
    from evaluation.dataset_audit import scan_images

    img = Image.new("RGB", (32, 32), (10, 20, 30))
    img.save(tmp_path / "x.jpg")
    img.save(tmp_path / "y.jpg")  # same pixels, re-saved
    (tmp_path / "z.jpg").write_bytes((tmp_path / "x.jpg").read_bytes())
    records = scan_images([tmp_path])
    hashes = [r["sha256"] for r in records]
    # x and z are byte-identical by construction
    assert hashes[0] == hashes[2]


def test_shipped_eval_set_audit_state():
    """The shipped artifacts must keep matching the Phase 3 findings."""
    from evaluation.dataset_audit import run_audit

    report = run_audit(phash_threshold=6)
    assert report["images_scanned"] == 35
    assert report["exact_duplicate_groups"] == {}
    assert report["annotations"]["total_boxes"] == 38
    assert report["annotations"]["issues"] == []
    assert report["manifest"]["mismatches"] == []
    # Contamination evidence must stay visible, not silently disappear:
    # 7 eval images come from a source dataset's train split.
    assert len(report["filename_train_split_hints"]) == 7
    # The ORIGINAL training data is absent; the audit must say so, not
    # imply clean. (Since Phase 5, data/ exists but holds only the
    # license-verified verified_v1 set — a different thing entirely.)
    assert report["training_side"]["original_training_data_present"] is False


def test_eval_set_status_label_present():
    status = ROOT / "tests" / "artifacts" / "EVAL_SET_STATUS.md"
    assert status.exists()
    text = status.read_text().lower()
    assert "exploratory" in text and "contaminated" in text


# ---------------------------------------------------------------------------
# License inventory
# ---------------------------------------------------------------------------

def test_license_inventory_covers_requirements():
    inv_path = ROOT / "docs" / "licenses_inventory.json"
    assert inv_path.exists(), "regenerate with evaluation/license_inventory.py"
    inv = json.loads(inv_path.read_text())
    by_name = {d["name"].lower().replace("_", "-"): d for d in inv["distributions"]}
    required = {"ultralytics", "streamlit", "opencv-python", "pillow",
                "pyyaml", "pandas", "imageio", "imageio-ffmpeg"}
    missing = required - set(by_name)
    assert not missing, f"license inventory missing: {missing}"


def test_ultralytics_agpl_is_recorded():
    inv = json.loads((ROOT / "docs" / "licenses_inventory.json").read_text())
    ultra = next(d for d in inv["distributions"] if d["name"] == "ultralytics")
    assert "AGPL" in ultra["license_declared"]
    # The audit doc must exist and must not claim the matter is settled.
    audit = (ROOT / "docs" / "LICENSE_AUDIT.md").read_text()
    assert "not legal advice" in audit.lower()
    assert "Decisions requiring the owner" in audit


# ---------------------------------------------------------------------------
# Product claims & deployment configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rel",
    ["app.py", "interceptor_sim.py", "video_interceptor_testbench.py"],
)
def test_no_guaranteed_outcome_claims_in_user_facing_code(rel):
    text = (ROOT / rel).read_text()
    for banned in ("CERTIFIED 100% Pk", "• 100% Pk", "[100% Pk]",
                   "100.0% Pk", "neutralization is guaranteed",
                   "Guaranteed Evasion Intercept",
                   # Phase 6: fabricated pre-populated "certification" results
                   "100% Lethal Neutralization", "CERTIFIED COMBAT",
                   "100% COMBAT CERTIFICATION"):
        assert banned not in text, f"{rel} still contains claim: {banned}"


def test_simulation_disclaimer_present_in_app():
    text = (ROOT / "app.py").read_text()
    assert "Simulation only" in text
    assert "fictional" in text.lower()


def test_streamlit_secure_defaults_enabled():
    cfg = (ROOT / ".streamlit" / "config.toml").read_text()
    assert "enableXsrfProtection = true" in cfg
    assert "enableCORS = true" in cfg


def test_phase3_docs_exist():
    for rel in ("docs/DATASET_PROVENANCE.md", "docs/LICENSE_AUDIT.md",
                "docs/PRODUCT_READINESS.md"):
        assert (ROOT / rel).exists(), rel
