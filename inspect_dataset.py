from pathlib import Path
from collections import Counter
import yaml

root = Path("data/synthetic_v1")
yaml_file = root / "dataset.yaml"

if not yaml_file.exists():
    raise SystemExit(f"Dataset YAML not found: {yaml_file}")

cfg = yaml.safe_load(yaml_file.read_text(encoding="utf-8-sig"))
names = cfg.get("names", {})
if isinstance(names, list):
    names = dict(enumerate(names))

print("\n=== DATASET CONFIG ===")
print(yaml_file.read_text(encoding="utf-8-sig"))
print("Classes:", names)

errors = []
class_counts = Counter()
image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

for split in ("train", "val", "test"):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = sorted(
        p for p in image_dir.glob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ) if image_dir.exists() else []

    print(f"\n=== {split.upper()} ===")
    print("Images:", len(images))
    print("Label directory exists:", label_dir.exists())

    if not images:
        print("No images in this split.")
        continue

    missing = 0
    labeled_objects = 0

    for img in images:
        label = label_dir / f"{img.stem}.txt"
        if not label.exists():
            missing += 1
            continue

        for line_no, line in enumerate(
            label.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{label}:{line_no}: expected 5 values")
                continue

            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(f"{label}:{line_no}: non-numeric value")
                continue

            if cls not in names:
                errors.append(f"{label}:{line_no}: unknown class {cls}")

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                errors.append(f"{label}:{line_no}: coordinate outside [0,1]")

            if w <= 0 or h <= 0:
                errors.append(f"{label}:{line_no}: non-positive box size")

            class_counts[cls] += 1
            labeled_objects += 1

    print("Images missing label files:", missing)
    print("Objects labeled:", labeled_objects)

print("\n=== SUMMARY ===")
print("Objects per class ID:", dict(class_counts))
print("Validation errors:", len(errors))
for err in errors[:30]:
    print("ERROR:", err)
if len(errors) > 30:
    print("Additional errors omitted:", len(errors) - 30)

print("Dataset inspection finished.")
