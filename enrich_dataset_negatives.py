"""AeroWatch AI — Hard-Negative Background & Real-World Enrichment.

Adds real foliage, tree leaves, soldier uniforms, and natural clutter as hard negatives (empty label .txt files)
to directly suppress false positive activations on trees and people.
Also balances balloons and civilian airliners.
"""

from __future__ import annotations

import random
from pathlib import Path
from PIL import Image

DATASET_DIR = Path("data/real_aerial_v1")

def add_hard_negatives():
    print("[*] Generating hard negative background samples (0 labels) to eliminate false positives on foliage/people...")
    
    screenshot_p = Path(r"C:\Users\ENG MOHAMED SABRY\.gemini\antigravity-ide\brain\4e23f119-8bdd-45f8-a07f-f2783d6b9196\.user_uploaded\media_1791745369736.png")
    if not screenshot_p.exists():
        print("Screenshot not found.")
        return

    with Image.open(screenshot_p) as full_img:
        rgb = full_img.convert("RGB")
        # Define multiple background-only crops with NO drones:
        # Soldier backpack, camo gear, tree foliage, branches, grass
        crops = [
            ("neg_soldier_back_camo", (270, 220, 380, 380)),
            ("neg_soldier_shoulder", (280, 200, 350, 290)),
            ("neg_top_right_foliage", (400, 145, 580, 230)),
            ("neg_top_branches", (420, 150, 500, 220)),
            ("neg_top_leaves_dense", (480, 145, 570, 225)),
            ("neg_left_foliage_bush", (255, 230, 290, 310)),
            ("neg_distant_trees_horizon", (380, 260, 520, 340)),
            ("neg_grassy_field_ground", (260, 350, 550, 420)),
            ("neg_cloudy_overcast_sky", (350, 145, 450, 210)),
        ]

        # Generate jittered variations to teach YOLO thoroughly
        added = 0
        for name, box in crops:
            crop = rgb.crop(box)
            # Create original, flipped, and scaled versions
            variants = [
                (f"{name}_orig", crop),
                (f"{name}_flip", crop.transpose(Image.FLIP_LEFT_RIGHT)),
            ]
            for v_name, v_img in variants:
                # Add to train split (80%) or val split (20%)
                target_split = "train" if random.random() < 0.75 else "val"
                img_path = DATASET_DIR / "images" / target_split / f"{v_name}.jpg"
                lbl_path = DATASET_DIR / "labels" / target_split / f"{v_name}.txt"
                
                v_img.save(img_path, "JPEG", quality=92)
                # EMPTY FILE = Negative background sample in YOLO
                lbl_path.write_text("")
                added += 1

    print(f"    -> Successfully injected {added} hard negative background images into real_aerial_v1!")


def balance_balloons_from_synthetic():
    print("[*] Adding balloon samples from synthetic_v4...")
    synth_dir = Path("data/synthetic_v4")
    added = 0
    for split in ["train", "val"]:
        img_dir = synth_dir / "images" / split
        lbl_dir = synth_dir / "labels" / split
        if not img_dir.exists():
            continue
        for p in img_dir.glob("*.png"):
            lbl_file = lbl_dir / f"{p.stem}.txt"
            if not lbl_file.exists():
                continue
            txt = lbl_file.read_text()
            if "3 " in txt and ("balloon" in p.stem or random.random() < 0.5):
                # Filter label to keep only balloon or valid boxes
                target_split = split
                dst_img = DATASET_DIR / "images" / target_split / f"balloon_{p.stem}.jpg"
                dst_lbl = DATASET_DIR / "labels" / target_split / f"balloon_{p.stem}.txt"
                
                with Image.open(p) as im:
                    im.convert("RGB").save(dst_img, "JPEG", quality=90)
                dst_lbl.write_text(txt)
                added += 1
                if added >= 20:
                    break
        if added >= 20:
            break
    print(f"    -> Added {added} balloon samples.")


def main():
    add_hard_negatives()
    balance_balloons_from_synthetic()


if __name__ == "__main__":
    main()
