"""AeroWatch AI — dataset provenance & contamination audit.

Reproducible, read-only audit of the labelled image artifacts in this
repository. It produces *evidence for manual review* — it never deletes
or modifies anything.

Checks performed:
- SHA-256 exact-duplicate detection across all scanned images.
- Perceptual-hash (pHash) near-duplicate detection: pairs whose Hamming
  distance is at or below a configurable threshold are reported as
  SUSPECTED near-duplicates for human review, never auto-removed.
- Image dimension / format inspection.
- Annotation audit: label-file presence, YOLO format validity, class-id
  range, duplicate annotation lines, and identical annotation sets
  shared by different images (a copy/paste smell).
- Manifest integrity: every entry in tests/artifacts/manifest.json is
  checked against the file on disk (SHA-256 and ground-truth count).
- Source / split tracking: manifest source names plus filename hints
  (e.g. a `_train` token in a filename indicates the image came from a
  source dataset's *train* split — a contamination risk flag when the
  file sits in an evaluation set).

The training side of the comparison is reported honestly: the builders
reference `data/real_aerial_*` trees that are gitignored and ABSENT
from this repository, so direct train-vs-eval image comparison is
impossible from here. The audit states that instead of implying a
clean result.

Usage:
    python evaluation/dataset_audit.py [--phash-threshold 6] \
        [--out evaluation/results/dataset_audit.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Perceptual hash (pHash, DCT-based, 64-bit)
# ---------------------------------------------------------------------------

def phash64(image_path: Path) -> int:
    """Classic pHash: 32x32 grayscale -> DCT -> low-freq 8x8 -> median bits."""
    with Image.open(image_path) as im:
        gray = im.convert("L").resize((32, 32), Image.LANCZOS)
    px = np.asarray(gray, dtype=np.float64)
    # 2D DCT-II via separable matrix multiply
    n = 32
    k = np.arange(n)
    dct_matrix = np.cos((2 * k[:, None] + 1) * k[None, :] * math.pi / (2 * n))
    dct = dct_matrix @ px @ dct_matrix.T
    low = dct[:8, :8]
    med = np.median(low[1:, :])  # exclude DC term dominance
    bits = (low > med).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_images(roots: list[Path]) -> list[dict]:
    records = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                rel_path = str(path.relative_to(PROJECT_ROOT))
            except ValueError:
                rel_path = str(path)
            try:
                with Image.open(path) as im:
                    w, h, fmt = im.size[0], im.size[1], im.format
                ph = phash64(path)
            except Exception as e:  # corrupt fixture etc.
                records.append(
                    {"path": rel_path, "sha256": digest,
                     "error": f"unreadable image: {e}"}
                )
                continue
            records.append(
                {
                    "path": rel_path,
                    "sha256": digest,
                    "width": w,
                    "height": h,
                    "format": fmt,
                    "phash": f"{ph:016x}",
                    "filename_split_hint": (
                        "train" if "_train" in path.stem or "train" == path.stem.split("_")[-1]
                        else None
                    ),
                }
            )
    return records


def audit_annotations(image_records: list[dict]) -> dict:
    issues: list[dict] = []
    annotation_sets: dict[str, list[str]] = {}
    total_boxes = 0
    for rec in image_records:
        if "error" in rec:
            continue
        img = PROJECT_ROOT / rec["path"]
        label = img.with_suffix(".txt")
        rel = rec["path"]
        if not label.exists():
            issues.append({"type": "missing_label", "image": rel})
            continue
        lines = [ln.strip() for ln in label.read_text().splitlines() if ln.strip()]
        seen: set[str] = set()
        boxes_here = []
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                issues.append({"type": "malformed_label_line", "image": rel, "line": ln})
                continue
            try:
                cid = int(float(parts[0]))
                vals = [float(v) for v in parts[1:5]]
            except ValueError:
                issues.append({"type": "malformed_label_line", "image": rel, "line": ln})
                continue
            if cid < 0 or cid > 4:
                issues.append({"type": "class_id_out_of_range", "image": rel, "line": ln})
            if any(v < 0.0 or v > 1.0 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                issues.append({"type": "box_out_of_bounds", "image": rel, "line": ln})
            if ln in seen:
                issues.append({"type": "duplicate_annotation_line", "image": rel, "line": ln})
            seen.add(ln)
            boxes_here.append(ln)
            total_boxes += 1
        if boxes_here:
            key = "||".join(sorted(boxes_here))
            annotation_sets.setdefault(key, []).append(rel)
    shared = {k: v for k, v in annotation_sets.items() if len(v) > 1}
    for _k, imgs in shared.items():
        issues.append({"type": "identical_annotation_set_across_images", "images": imgs})
    return {"issues": issues, "total_boxes": total_boxes}


def audit_manifest() -> dict:
    manifest_path = PROJECT_ROOT / "tests" / "artifacts" / "manifest.json"
    if not manifest_path.exists():
        return {"present": False}
    manifest = json.loads(manifest_path.read_text())
    mismatches = []
    entries = manifest.get("images", [])
    sources: dict[str, int] = {}
    licenses: dict[str, int] = {}
    for entry in entries:
        rel = entry.get("relative_path", "")
        sources[entry.get("source_name", "?")] = sources.get(entry.get("source_name", "?"), 0) + 1
        licenses[entry.get("license", "?")] = licenses.get(entry.get("license", "?"), 0) + 1
        f = PROJECT_ROOT / rel
        if not f.exists():
            mismatches.append({"image": rel, "problem": "file missing"})
            continue
        actual = hashlib.sha256(f.read_bytes()).hexdigest()
        if entry.get("sha256") and actual != entry["sha256"]:
            mismatches.append({"image": rel, "problem": "sha256 mismatch vs manifest"})
        label = f.with_suffix(".txt")
        gt_lines = (
            len([ln for ln in label.read_text().splitlines() if ln.strip()])
            if label.exists()
            else 0
        )
        if entry.get("ground_truth_count") is not None and gt_lines != entry["ground_truth_count"]:
            mismatches.append(
                {"image": rel, "problem": f"label lines {gt_lines} != manifest GT count {entry['ground_truth_count']}"}
            )
    return {
        "present": True,
        "entries": len(entries),
        "mismatches": mismatches,
        "sources": sources,
        "licenses": licenses,
    }


def run_audit(phash_threshold: int = 6) -> dict:
    roots = [
        PROJECT_ROOT / "tests" / "artifacts" / "images",
        PROJECT_ROOT / "tests" / "artifacts" / "negative_cases",
    ]
    records = scan_images(roots)

    # Exact duplicates
    by_hash: dict[str, list[str]] = {}
    for r in records:
        by_hash.setdefault(r["sha256"], []).append(r["path"])
    exact_dupes = {h: ps for h, ps in by_hash.items() if len(ps) > 1}

    # Near duplicates (pHash), skipping exact-identical pairs
    readable = [r for r in records if "phash" in r]
    near_dupes = []
    for i in range(len(readable)):
        for j in range(i + 1, len(readable)):
            a, b = readable[i], readable[j]
            if a["sha256"] == b["sha256"]:
                continue
            d = hamming(int(a["phash"], 16), int(b["phash"], 16))
            if d <= phash_threshold:
                near_dupes.append(
                    {"a": a["path"], "b": b["path"], "hamming_distance": d,
                     "note": "SUSPECTED near-duplicate — manual review required, do not auto-delete"}
                )

    training_side = PROJECT_ROOT / "data"
    return {
        "images_scanned": len(records),
        "unreadable": [r["path"] for r in records if "error" in r],
        "exact_duplicate_groups": exact_dupes,
        "near_duplicate_pairs": near_dupes,
        "phash_threshold": phash_threshold,
        "filename_train_split_hints": [
            r["path"] for r in records if r.get("filename_split_hint") == "train"
        ],
        "annotations": audit_annotations(records),
        "manifest": audit_manifest(),
        "training_side": {
            "data_dir_present": training_side.exists(),
            "note": (
                "Training datasets (data/real_aerial_*) are gitignored and "
                "absent from this repository, so direct train-vs-eval image "
                "comparison is IMPOSSIBLE from the repo alone. Overlap can "
                "only be assessed at source level: the builders pull from "
                "the same Hugging Face datasets that supplied these "
                "evaluation artifacts (see docs/DATASET_PROVENANCE.md)."
            ),
        },
        "records": records,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phash-threshold", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = run_audit(args.phash_threshold)
    out = args.out or str(
        PROJECT_ROOT / "evaluation" / "results" / "dataset_audit.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))

    print(f"images scanned: {report['images_scanned']}")
    print(f"exact duplicate groups: {len(report['exact_duplicate_groups'])}")
    for h, ps in report["exact_duplicate_groups"].items():
        print(f"  sha256 {h[:12]}… -> {ps}")
    print(f"near-duplicate pairs (pHash <= {args.phash_threshold}): {len(report['near_duplicate_pairs'])}")
    for nd in report["near_duplicate_pairs"]:
        print(f"  d={nd['hamming_distance']}  {nd['a']}  <->  {nd['b']}")
    print(f"filename train-split hints: {report['filename_train_split_hints']}")
    ann = report["annotations"]
    print(f"annotation boxes: {ann['total_boxes']}, issues: {len(ann['issues'])}")
    for issue in ann["issues"]:
        print(f"  {issue}")
    man = report["manifest"]
    print(f"manifest: present={man.get('present')} entries={man.get('entries')} mismatches={len(man.get('mismatches', []))}")
    for m in man.get("mismatches", []):
        print(f"  {m}")
    print(f"training side present: {report['training_side']['data_dir_present']} (see report note)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
