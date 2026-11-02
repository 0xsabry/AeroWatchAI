"""Phase 5 guards for data/verified_v1 (the frozen, license-verified set).

These tests protect the properties the set was built for: provenance
on every image, no exact-duplicate leakage across splits, no overlap
with the historically contaminated tests/artifacts set, and a test
split that stays frozen at its recorded size.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "verified_v1"
ALLOWED_LICENSE_PREFIXES = ("CC0", "Public domain", "CC BY", "CC BY-SA")


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not (DATA / "MANIFEST.json").exists():
        pytest.skip("verified_v1 dataset not present")
    return json.loads((DATA / "MANIFEST.json").read_text())


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_every_image_has_verified_license(manifest):
    for img in manifest["images"]:
        assert img["license"], f"no license recorded for {img['file']}"
        assert img["license"].startswith(ALLOWED_LICENSE_PREFIXES), img["license"]
        assert img["source_page"] and img["source_page"].startswith("https://commons.wikimedia.org/")


def test_manifest_hashes_match_files(manifest):
    for img in manifest["images"]:
        p = DATA / img["file"]
        assert p.exists(), f"missing {img['file']}"
        assert _sha(p) == img["sha256"], f"hash drift in {img['file']}"


def test_no_exact_duplicates_across_splits(manifest):
    by_hash: dict[str, str] = {}
    for img in manifest["images"]:
        split = img["split"].replace("negative_", "")
        if img["sha256"] in by_hash:
            assert by_hash[img["sha256"]] == split, (
                f"same image in two splits: {img['file']}"
            )
        by_hash[img["sha256"]] = split


def test_no_overlap_with_legacy_eval_set(manifest):
    legacy = {
        _sha(p)
        for p in (ROOT / "tests" / "artifacts" / "images").glob("*.jpg")
    }
    overlap = [i["file"] for i in manifest["images"] if i["sha256"] in legacy]
    assert not overlap, f"verified set overlaps legacy artifacts: {overlap}"


def test_frozen_test_split_size(manifest):
    counts = manifest["counts"]
    assert counts["test"] == 46
    assert counts["negative_test"] == 15
