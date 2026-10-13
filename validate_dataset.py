"""AeroWatch AI — Strict Dataset Validator.

Validates YOLO dataset integrity, annotations, coordinate bounds, class consistency,
and train/val/test split separation.

Exit code:
  0 = Dataset valid
  1 = Validation errors or insufficient sample count (unless --draft-mode)
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

VALID_SCHEMAS = [
    # 4-class civilian
    {0: ("drone",), 1: ("aircraft", "airplane"), 2: ("bird",), 3: ("balloon",)},
    # 5-class with warplanes / military_jet
    {0: ("drone",), 1: ("aircraft", "airplane"), 2: ("bird",), 3: ("balloon",), 4: ("military_jet",)},
    # 6-class with separated kamikaze/loitering drones
    {0: ("drone",), 1: ("aircraft", "airplane"), 2: ("bird",), 3: ("balloon",), 4: ("military_jet",), 5: ("kamikaze_drone",)},
]

CANONICAL_CLASSES = {
    0: "drone",
    1: "aircraft",
    2: "bird",
    3: "balloon",
    4: "military_jet",
    5: "kamikaze_drone",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Strict YOLO Dataset Validator for AeroWatch AI")
    parser.add_argument("--data", default="data/synthetic_v3/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--min-train", type=int, default=100, help="Minimum required training images")
    parser.add_argument("--min-val", type=int, default=20, help="Minimum required validation images")
    parser.add_argument("--min-test", type=int, default=10, help="Minimum required test images")
    parser.add_argument("--draft-mode", action="store_true", help="Allow smaller datasets to pass for pipeline smoke testing only")
    parser.add_argument("--strict-names", action="store_true", default=True, help="Enforce canonical AeroWatch class names and IDs")
    return parser.parse_args()


def load_dataset_yaml(yaml_path: Path) -> tuple[dict, Path, dict[int, str]]:
    if not yaml_path.is_file():
        raise FileNotFoundError(f"dataset.yaml not found: {yaml_path}")

    raw_cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
    root_val = raw_cfg.get("path", str(yaml_path.parent))
    root_path = Path(root_val)
    if not root_path.is_absolute():
        root_path = (yaml_path.parent / root_path).resolve()

    names_raw = raw_cfg.get("names", {})
    if isinstance(names_raw, list):
        names = {i: name for i, name in enumerate(names_raw)}
    elif isinstance(names_raw, dict):
        names = {int(k): str(v) for k, v in names_raw.items()}
    else:
        raise ValueError("'names' in dataset.yaml must be a dictionary or list")

    return raw_cfg, root_path, names


def resolve_split_path(root: Path, raw_cfg: dict, split: str) -> Path | None:
    val = raw_cfg.get(split)
    if not val:
        return None
    p = Path(val)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def validate_dataset(args) -> bool:
    yaml_path = Path(args.data).resolve()
    print("=" * 65)
    print("AEROWATCH AI — STRICT DATASET VALIDATION REPORT")
    print(f"Config path: {yaml_path}")

    errors: list[str] = []
    warnings: list[str] = []

    # 1. Load and verify dataset.yaml
    try:
        cfg, root_path, names = load_dataset_yaml(yaml_path)
    except Exception as e:
        print(f"[FATAL] Failed to read dataset.yaml: {e}")
        return False

    print(f"Dataset root: {root_path}")
    print(f"Classes defined: {names}")

    # Canonical class check
    if args.strict_names:
        matching_schema = None
        for schema in VALID_SCHEMAS:
            if set(names.keys()) == set(schema.keys()):
                if all(names[k] in schema[k] for k in schema):
                    matching_schema = schema
                    break
        if matching_schema is None:
            errors.append(f"Classes {names} do not match any recognized AeroWatch schema (4-class civilian, 5-class military_jet, or 6-class kamikaze_drone)")

    # Prohibited weapon/munition keywords check
    forbidden = {"missile", "warhead", "bomb", "explosive"}
    for cid, name in names.items():
        if any(f in name.lower() for f in forbidden):
            errors.append(f"Prohibited weapon/munition class detected: ID {cid} = '{name}'")

    # 2. Inspect splits
    splits_to_check = ["train", "val", "test"]
    split_image_counts: dict[str, int] = {}
    split_class_counts: dict[str, Counter] = {}
    seen_stems: dict[str, str] = {}  # stem -> split for duplicate detection

    for split in splits_to_check:
        split_dir = resolve_split_path(root_path, cfg, split)
        print(f"\n--- Checking split: [{split.upper()}] ---")

        if split_dir is None:
            if split in ("train", "val"):
                errors.append(f"Required split '{split}' is missing in dataset.yaml")
            else:
                warnings.append(f"Optional split '{split}' is not specified in dataset.yaml")
            split_image_counts[split] = 0
            continue

        if not split_dir.is_dir():
            if split in ("train", "val"):
                errors.append(f"Directory for '{split}' does not exist: {split_dir}")
            else:
                warnings.append(f"Directory for optional '{split}' does not exist: {split_dir}")
            split_image_counts[split] = 0
            continue

        # In standard YOLO layout, labels directory mirrors images directory:
        # e.g., root/images/train -> root/labels/train
        label_dir = root_path / "labels" / split
        if not label_dir.is_dir():
            errors.append(f"Labels directory for '{split}' does not exist: {label_dir}")

        images = sorted(
            p for p in split_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        split_image_counts[split] = len(images)
        split_class_counts[split] = Counter()
        images_with_empty_labels = 0
        corrupted_images = 0

        print(f"Images found: {len(images)}")
        print(f"Labels dir  : {label_dir} (exists: {label_dir.is_dir()})")

        for img in images:
            # Check duplicate filenames across splits
            if img.stem in seen_stems:
                errors.append(f"Duplicate image stem '{img.stem}' found in '{split}' and '{seen_stems[img.stem]}'")
            else:
                seen_stems[img.stem] = split

            # Verify image readability
            try:
                with Image.open(img) as pil_img:
                    pil_img.verify()
            except Exception as e:
                errors.append(f"Corrupted or unreadable image: {img} ({e})")
                corrupted_images += 1
                continue

            # Check matching label file
            lbl_file = label_dir / f"{img.stem}.txt"
            if not lbl_file.is_file():
                errors.append(f"Missing label file for image: {img.name} -> {lbl_file}")
                continue

            # Parse label file
            lines = [l.strip() for l in lbl_file.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
            if not lines:
                images_with_empty_labels += 1

            for line_no, line in enumerate(lines, start=1):
                tokens = line.split()
                if len(tokens) != 5:
                    errors.append(f"{lbl_file.name}:{line_no}: Expected 5 fields, found {len(tokens)}")
                    continue

                try:
                    cls_id = int(tokens[0])
                    x, y, w, h = map(float, tokens[1:])
                except ValueError:
                    errors.append(f"{lbl_file.name}:{line_no}: Non-numeric bounding box values")
                    continue

                if cls_id not in names:
                    errors.append(f"{lbl_file.name}:{line_no}: Class ID {cls_id} not defined in dataset.yaml")

                if not all(math.isfinite(v) for v in (x, y, w, h)):
                    errors.append(f"{lbl_file.name}:{line_no}: Non-finite coordinate value")
                    continue

                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    errors.append(f"{lbl_file.name}:{line_no}: Center coordinates ({x:.4f}, {y:.4f}) outside [0, 1]")

                if w <= 0.0 or h <= 0.0:
                    errors.append(f"{lbl_file.name}:{line_no}: Non-positive box dimensions (w={w}, h={h})")

                if x + w / 2.0 > 1.05 or x - w / 2.0 < -0.05 or y + h / 2.0 > 1.05 or y - h / 2.0 < -0.05:
                    warnings.append(f"{lbl_file.name}:{line_no}: Bounding box exceeds frame significantly")

                split_class_counts[split][cls_id] += 1

        print(f"Empty labels : {images_with_empty_labels}")
        print(f"Class counts : {dict(split_class_counts[split])}")

    # 3. Overall Dataset Distribution and Requirements
    total_images = sum(split_image_counts.values())
    total_class_counts = Counter()
    for sc in split_class_counts.values():
        total_class_counts.update(sc)

    print("\n" + "=" * 65)
    print("DATASET AGGREGATE SUMMARY")
    print(f"Total images      : {total_images} (Train: {split_image_counts.get('train', 0)}, Val: {split_image_counts.get('val', 0)}, Test: {split_image_counts.get('test', 0)})")
    print("Class detections  :")
    for cid, cname in names.items():
        count = total_class_counts[cid]
        print(f"  [{cid}] {cname:10s} : {count} instances")

    # Check class representation
    for cid, cname in names.items():
        if total_class_counts[cid] == 0:
            if args.draft_mode:
                warnings.append(f"Class '{cname}' (ID {cid}) has 0 instances in this draft sample set (expected to be populated in production runs)")
            else:
                errors.append(f"Class '{cname}' (ID {cid}) has 0 instances across the entire dataset!")

    # 4. Minimum sample thresholds check
    train_count = split_image_counts.get("train", 0)
    val_count = split_image_counts.get("val", 0)
    test_count = split_image_counts.get("test", 0)

    if not args.draft_mode:
        if train_count < args.min_train:
            errors.append(f"Insufficient training samples: found {train_count}, required at least {args.min_train}")
        if val_count < args.min_val:
            errors.append(f"Insufficient validation samples: found {val_count}, required at least {args.min_val}")
        if test_count < args.min_test:
            warnings.append(f"Test split has {test_count} samples (recommended >= {args.min_test})")
    else:
        print("\n[NOTE: DRAFT MODE ACTIVE]")
        if train_count < args.min_train or val_count < args.min_val:
            warnings.append(
                f"Draft dataset has only {train_count} train / {val_count} val images. "
                f"Sufficient for pipeline validation ONLY; NOT viable for training a useful model."
            )

    # 5. Final Verdict
    print("\n" + "=" * 65)
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings[:15]:
            print(f"  [!] {w}")
        if len(warnings) > 15:
            print(f"  ... and {len(warnings) - 15} more warnings.")

    if errors:
        print(f"\nVALIDATION FAILED ({len(errors)} error(s)):")
        for err in errors[:25]:
            print(f"  [X] {err}")
        if len(errors) > 25:
            print(f"  ... and {len(errors) - 25} more errors.")
        print("\nAction required: Fix reported errors or regenerate dataset.")
        return False
    else:
        print("\n>>> VALIDATION PASSED: All structural, geometric, and label checks succeeded. <<<")
        if args.draft_mode and (train_count < args.min_train or val_count < args.min_val):
            print(">>> STATUS: DRAFT READY (Use for code/pipeline tests only, not production training) <<<")
        else:
            print(">>> STATUS: FULL DATASET READY FOR YOLO TRAINING <<<")
        return True


def main():
    args = parse_arguments()
    is_valid = validate_dataset(args)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
