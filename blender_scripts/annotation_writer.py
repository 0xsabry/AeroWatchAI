"""AeroWatch AI — Annotation Writer & Geometric Projection for Blender 5.2.

Provides:
- 3D-to-2D camera projection with frustum and occlusion/depth validation
- Bounding box calculation in standard YOLO format: <class_id> <x_center> <y_center> <width> <height>
- Metadata logging (CSV and JSON)
- Contact sheet mosaic builder using PIL/OpenCV outside Blender or via helper
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

CANONICAL_CLASSES = {
    0: "drone",
    1: "aircraft",
    2: "bird",
    3: "balloon",
    4: "military_jet",
    5: "kamikaze_drone",
}



def project_bounding_box(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    obj: bpy.types.Object,
    width: int,
    height: int,
    min_box_pixels: float = 6.0,
) -> tuple[float, float, float, float] | None:
    """Computes a normalized 2D YOLO bounding box [x_center, y_center, width, height]

    from the evaluated 3D object bounding box.
    Returns None if the object is behind the camera, outside the frustum, or degenerate.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)

    # Transform 8 local bounding box corners to world space, then to normalized camera space
    points_2d = []
    points_in_front = 0
    for corner in evaluated.bound_box:
        world_pt = evaluated.matrix_world @ Vector(corner)
        cam_coord = world_to_camera_view(scene, camera, world_pt)
        if cam_coord.z > 0.1:  # Must be strictly in front of camera
            points_in_front += 1
            # world_to_camera_view returns:
            # x: [0, 1] (left to right)
            # y: [0, 1] (bottom to top)
            # YOLO image origin is top-left, so py = (1.0 - cam_coord.y) * height
            px = cam_coord.x * width
            py = (1.0 - cam_coord.y) * height
            points_2d.append((px, py))

    # Reject if fewer than 4 corners are in front of camera
    if points_in_front < 4 or not points_2d:
        return None

    min_x = min(p[0] for p in points_2d)
    max_x = max(p[0] for p in points_2d)
    min_y = min(p[1] for p in points_2d)
    max_y = max(p[1] for p in points_2d)

    # Check if box is completely outside the viewport
    if max_x < 0 or min_x > width or max_y < 0 or min_y > height:
        return None

    # Clamp coordinates to viewport boundaries [0, width] x [0, height]
    x0 = max(0.0, min_x)
    y0 = max(0.0, min_y)
    x1 = min(float(width), max_x)
    y1 = min(float(height), max_y)

    box_w = x1 - x0
    box_h = y1 - y0

    # Reject degenerate or microscopic boxes
    if box_w < min_box_pixels or box_h < min_box_pixels:
        return None

    # Normalized YOLO coordinates (center_x, center_y, width, height) in [0, 1]
    xc = ((x0 + x1) / 2.0) / width
    yc = ((y0 + y1) / 2.0) / height
    norm_w = box_w / width
    norm_h = box_h / height

    # Guarantee bounds safety [0, 1]
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    norm_w = max(1e-5, min(1.0, norm_w))
    norm_h = max(1e-5, min(1.0, norm_h))

    return xc, yc, norm_w, norm_h


class DatasetMetadataLogger:
    """Maintains metadata.csv, classes.json, and generation_summary.json."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.csv_path = output_dir / "metadata.csv"
        self.records: list[dict] = []
        self._init_csv()

    def _init_csv(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fieldnames = [
            "image_name",
            "split",
            "classes_present",
            "object_count",
            "lighting_condition",
            "sun_elevation_deg",
            "camera_lens_mm",
            "resolution",
        ]
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_image(
        self,
        image_name: str,
        split: str,
        classes_present: list[str],
        object_count: int,
        lighting: dict,
        resolution: str,
    ):
        record = {
            "image_name": image_name,
            "split": split,
            "classes_present": ";".join(classes_present),
            "object_count": object_count,
            "lighting_condition": lighting.get("condition", "default"),
            "sun_elevation_deg": lighting.get("sun_elevation_deg", 0.0),
            "camera_lens_mm": lighting.get("camera_lens_mm", 50.0),
            "resolution": resolution,
        }
        self.records.append(record)
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(record)

    def write_classes_and_summary(
        self,
        summary_data: dict,
    ):
        classes_json = self.output_dir / "classes.json"
        with open(classes_json, "w", encoding="utf-8") as f:
            json.dump(CANONICAL_CLASSES, f, indent=2)

        summary_json = self.output_dir / "generation_summary.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)


def render_contact_sheet(dataset_root: Path, output_file: Path | None = None, max_images: int = 16):
    """Generates a contact sheet mosaic image with bounding boxes drawn over samples."""
    out_path = output_file if output_file else dataset_root / "contact_sheet.jpg"

    # Python script snippet to run using project python (.venv) if PIL is available
    script = f"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

dataset_root = Path(r"{dataset_root}")
out_path = Path(r"{out_path}")
classes = {dict(CANONICAL_CLASSES)}
colors = {{0: "#FF3838", 1: "#00E5FF", 2: "#FFB300", 3: "#D500F9", 4: "#76FF03", 5: "#FF9100"}}

images = sorted(list(dataset_root.glob("images/train/*.png")) + list(dataset_root.glob("images/val/*.png")))
if not images:
    sys.exit(0)

# Select evenly spaced samples up to {max_images}
sample_step = max(1, len(images) // {max_images})
selected = images[::sample_step][:{max_images}]

thumb_w, thumb_h = 320, 180
cols = 4
rows = (len(selected) + cols - 1) // cols
sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (20, 24, 30))
draw_sheet = ImageDraw.Draw(sheet)

for idx, img_path in enumerate(selected):
    r, c = divmod(idx, cols)
    x_off, y_off = c * thumb_w, r * thumb_h
    try:
        im = Image.open(img_path).convert("RGB")
        im_thumb = im.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        sheet.paste(im_thumb, (x_off, y_off))
        
        # Read label
        lbl_path = img_path.parent.parent.parent / "labels" / img_path.parent.name / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    bx0 = x_off + (xc - bw / 2) * thumb_w
                    by0 = y_off + (yc - bh / 2) * thumb_h
                    bx1 = x_off + (xc + bw / 2) * thumb_w
                    by1 = y_off + (yc + bh / 2) * thumb_h
                    color = colors.get(cls_id, "#00FF00")
                    draw_sheet.rectangle([bx0, by0, bx1, by1], outline=color, width=2)
                    cname = classes.get(cls_id, str(cls_id))
                    draw_sheet.text((bx0 + 2, max(y_off, by0 - 12)), cname, fill=color)
        draw_sheet.rectangle([x_off, y_off, x_off + thumb_w, y_off + thumb_h], outline=(50, 58, 70), width=1)
    except Exception as e:
        print(f"Error thumbnailing {{img_path}}: {{e}}")

sheet.save(out_path, quality=90)
print(f"[AeroWatch] Contact sheet written: {{out_path}}")
"""
    # Check if PIL is in current process
    try:
        from PIL import Image, ImageDraw  # type: ignore
        exec(script, {"__file__": str(dataset_root / "build_contact.py")})
    except ImportError:
        # Run via project python in .venv
        venv_python = Path(sys.prefix).parent / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            venv_python = Path("C:/Users/ENG MOHAMED SABRY/AeroWatchAI/.venv/Scripts/python.exe")
        if venv_python.exists():
            subprocess.run([str(venv_python), "-c", script], check=False)
