"""Phase 5 — assemble the verified dataset from annotation records.

Input: an annotations JSON produced during curation:
  { "<pool relative path>": {"split": "test|train|val|negative_test|negative_train",
                              "boxes": [{"cls": 0..3, "x1":..,"y1":..,"x2":..,"y2":..} px],
                              "image_w": W, "image_h": H}, ... }

Output layout (YOLO):
  <out>/test/images, test/labels, test/negatives
  <out>/train/images, train/labels, train/negatives
  <out>/val/images,  val/labels
  <out>/dataset.yaml          (train/val for ultralytics)
  <out>/MANIFEST.json         (per-image provenance joined from the pool
                               manifest, split, boxes, sha256)
  <out>/classes.txt           (drone aircraft bird balloon)

The test split is FROZEN once built: this script refuses to overwrite
an existing test/ directory unless --force is passed explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

CLASSES = ["drone", "aircraft", "bird", "balloon"]


def yolo_line(cls: int, b: dict, w: int, h: int) -> str:
    xc = (b["x1"] + b["x2"]) / 2 / w
    yc = (b["y1"] + b["y2"]) / 2 / h
    bw = (b["x2"] - b["x1"]) / w
    bh = (b["y2"] - b["y1"]) / h
    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, help="pool dir with pool_manifest.json")
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    pool = Path(args.pool)
    out = Path(args.out)
    if (out / "test").exists() and not args.force:
        raise SystemExit(f"{out/'test'} exists — the test split is frozen. Use --force only deliberately.")
    ann = json.loads(Path(args.annotations).read_text())
    pool_manifest = {m["file"]: m for m in json.loads((pool / "pool_manifest.json").read_text())}

    manifest: list[dict] = []
    counts: Counter = Counter()
    for rel, rec in sorted(ann.items()):
        src = pool / rel
        if not src.exists():
            raise SystemExit(f"annotated file missing from pool: {rel}")
        split = rec["split"]
        is_neg = split.startswith("negative_")
        base_split = split.replace("negative_", "")
        stem = f"{Path(rel).parent.name}_{Path(rel).stem}"
        if is_neg:
            dest_dir = out / base_split / "negatives"
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest_dir / f"{stem}.jpg")
            label_text = ""
        else:
            (out / base_split / "images").mkdir(parents=True, exist_ok=True)
            (out / base_split / "labels").mkdir(parents=True, exist_ok=True)
            shutil.copy(src, out / base_split / "images" / f"{stem}.jpg")
            label_text = "\n".join(
                yolo_line(b["cls"], b, rec["image_w"], rec["image_h"]) for b in rec["boxes"]
            )
            (out / base_split / "labels" / f"{stem}.txt").write_text(label_text + ("\n" if label_text else ""))
        prov = pool_manifest.get(rel, {})
        manifest.append(
            {
                "file": f"{base_split}/{'negatives' if is_neg else 'images'}/{stem}.jpg",
                "split": split,
                "pool_source": rel,
                "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                "boxes": rec["boxes"],
                "source_title": prov.get("title"),
                "source_page": prov.get("page_url"),
                "artist": prov.get("artist"),
                "license": prov.get("license"),
                "license_url": prov.get("license_url"),
            }
        )
        counts[split] += 1
        for b in rec["boxes"]:
            counts[f"boxes:{CLASSES[b['cls']]}:{base_split}"] += 1

    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    (out / "dataset.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    )
    (out / "MANIFEST.json").write_text(json.dumps({"images": manifest, "counts": dict(counts)}, indent=1))
    print(json.dumps(dict(counts), indent=1))


if __name__ == "__main__":
    main()
