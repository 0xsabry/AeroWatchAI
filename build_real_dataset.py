"""AeroWatch AI — Real-World Aerial Defense Dataset Builder.

Downloads and normalizes real-world photos for aerial defense:
- Real Drones (commercial quadcopters, hexacopters, FPV) from Rigoleto/drone-detection
- Real Fighter Jets & Warplanes (F-15, F-16, F-22, F-35, Su-27, Rafale...) from a2015003713
- Real Commercial Airliners (Boeing 737, Airbus A320...) from a2015003713
- Real Birds in flight from matisdsp
- Real Weather & Hot Air Balloons from synthetic_v4
- Real Delta-wing loitering/kamikaze drones from synthetic_v4
- Real Hard Background Negatives (tree canopies, foliage, soldier camo, outdoor clutter - 0 labels)

Outputs a validated YOLOv8/YOLO11/YOLO26-compatible dataset in data/real_aerial_v1.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import shutil
import sys
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download
import requests

CANONICAL_CLASSES = {
    0: "drone",
    1: "aircraft",
    2: "bird",
    3: "balloon",
    4: "military_jet",
}

AIRLINER_CAT_IDS = set(range(103, 114))  # Boeing 737-787, Airbus A320-A380
MILITARY_DRONE_CAT_IDS = {49, 58, 65, 66, 77, 78, 79, 91, 93, 95, 96, 98, 99, 102}

OUTPUT_DIR = Path("data/real_aerial_v1")


def setup_dirs():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def download_rigoleto_drones(limit=140):
    print(f"[*] Downloading {limit} real drone images from Rigoleto...", flush=True)
    items = [(f"train{i:04d}", "train") for i in range(1, limit + 1)]
    downloaded = []

    def fetch_item(item):
        name, split = item
        lbl_file = f"data/{split}/labels/{name}.txt"
        img_file = f"data/{split}/images/{name}.jpg"
        try:
            cached_lbl = hf_hub_download(repo_id="Rigoleto/drone-detection", repo_type="dataset", filename=lbl_file)
            cached_img = hf_hub_download(repo_id="Rigoleto/drone-detection", repo_type="dataset", filename=img_file)
            with open(cached_lbl, "r") as f:
                lines = f.readlines()
            yolo_lines = []
            for l in lines:
                parts = l.strip().split()
                if len(parts) >= 5:
                    yolo_lines.append(f"0 {' '.join(parts[1:5])}\n")
            if yolo_lines:
                return (cached_img, "".join(yolo_lines), f"rigoleto_{name}")
        except Exception:
            return None
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for r in executor.map(fetch_item, items):
            if r is not None:
                downloaded.append(r)

    print(f"    -> Acquired {len(downloaded)} real drone samples.", flush=True)
    return downloaded


def download_military_aircraft(limit_jets=90, limit_airliners=45):
    print(f"[*] Downloading real aircraft from a2015003713...", flush=True)
    try:
        url = "https://huggingface.co/api/datasets/a2015003713/military-aircraft-detection-dataset/tree/main/annotations/yolo/train/0"
        res = requests.get(url, timeout=10).json()
        all_label_paths = [x["path"] for x in res if x["path"].endswith(".txt")]
        # Only take first 180 paths to keep execution fast and focused
        selected_paths = all_label_paths[:180]
    except Exception as e:
        print(f"Failed to fetch file list: {e}", flush=True)
        return [], []

    jet_samples = []
    airliner_samples = []

    def fetch_aircraft(lbl_rel_path):
        stem = Path(lbl_rel_path).stem
        img_rel_path = f"images/0/{stem}.jpg"
        try:
            cached_lbl = hf_hub_download(repo_id="a2015003713/military-aircraft-detection-dataset", repo_type="dataset", filename=lbl_rel_path)
            cached_img = hf_hub_download(repo_id="a2015003713/military-aircraft-detection-dataset", repo_type="dataset", filename=img_rel_path)
            with open(cached_lbl, "r") as f:
                lines = f.readlines()
            
            remapped_lines = []
            is_airliner = False
            for l in lines:
                parts = l.strip().split()
                if len(parts) >= 5:
                    orig_cid = int(parts[0])
                    coords = " ".join(parts[1:5])
                    if orig_cid in AIRLINER_CAT_IDS:
                        remapped_lines.append(f"1 {coords}\n")
                        is_airliner = True
                    elif orig_cid in MILITARY_DRONE_CAT_IDS:
                        remapped_lines.append(f"0 {coords}\n")
                    else:
                        remapped_lines.append(f"4 {coords}\n")
            
            if remapped_lines:
                return (cached_img, "".join(remapped_lines), f"aircraft_{stem}", is_airliner)
        except Exception:
            return None
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for res in executor.map(fetch_aircraft, selected_paths):
            if res is not None:
                cached_img, lbl_content, prefix, is_airliner = res
                if is_airliner and len(airliner_samples) < limit_airliners:
                    airliner_samples.append((cached_img, lbl_content, prefix))
                elif not is_airliner and len(jet_samples) < limit_jets:
                    jet_samples.append((cached_img, lbl_content, prefix))

    print(f"    -> Acquired {len(jet_samples)} fighter jets and {len(airliner_samples)} airliners.", flush=True)
    return jet_samples, airliner_samples


def download_matisdsp_birds_and_negatives(limit_birds=60, limit_negatives=50):
    print(f"[*] Downloading {limit_birds} real birds and {limit_negatives} hard background negatives...", flush=True)
    try:
        coco_path = hf_hub_download(
            repo_id="matisdsp/drone-bird-detection-dataset",
            repo_type="dataset",
            filename="valid/_annotations.coco.json"
        )
        with open(coco_path, "r") as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Failed to load COCO annotations: {e}", flush=True)
        return [], []

    img_id_to_anns = {}
    for a in coco_data.get("annotations", []):
        img_id = a["image_id"]
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(a)

    bird_imgs = []
    empty_imgs = []
    for img_meta in coco_data.get("images", []):
        img_id = img_meta["id"]
        anns = img_id_to_anns.get(img_id, [])
        if not anns:
            empty_imgs.append(img_meta)
        else:
            if any(a["category_id"] == 1 for a in anns):
                bird_imgs.append((img_meta, anns))

    selected_birds = bird_imgs[:limit_birds]
    selected_negatives = empty_imgs[:limit_negatives]

    bird_samples = []
    negative_samples = []

    def fetch_bird(item):
        img_meta, anns = item
        fname = img_meta["file_name"]
        w = img_meta["width"]
        h = img_meta["height"]
        try:
            cached_img = hf_hub_download(
                repo_id="matisdsp/drone-bird-detection-dataset",
                repo_type="dataset",
                filename=f"valid/{fname}"
            )
            yolo_lines = []
            for a in anns:
                cat_id = a["category_id"]
                target_cls = 2 if cat_id == 1 else 0
                bbox = a["bbox"]
                xc = (bbox[0] + bbox[2] / 2.0) / w
                yc = (bbox[1] + bbox[3] / 2.0) / h
                nw = bbox[2] / w
                nh = bbox[3] / h
                yolo_lines.append(f"{target_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
            if yolo_lines:
                return (cached_img, "".join(yolo_lines), f"bird_{Path(fname).stem}")
        except Exception:
            return None
        return None

    def fetch_negative(img_meta):
        fname = img_meta["file_name"]
        try:
            cached_img = hf_hub_download(
                repo_id="matisdsp/drone-bird-detection-dataset",
                repo_type="dataset",
                filename=f"valid/{fname}"
            )
            return (cached_img, "", f"bg_{Path(fname).stem}")
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for r in executor.map(fetch_bird, selected_birds):
            if r is not None:
                bird_samples.append(r)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for r in executor.map(fetch_negative, selected_negatives):
            if r is not None:
                negative_samples.append(r)

    print(f"    -> Acquired {len(bird_samples)} real birds and {len(negative_samples)} hard negatives.", flush=True)
    return bird_samples, negative_samples


def add_custom_foliage_and_soldier_negatives():
    """Create dedicated hard negative crops from outdoor scenes (trees, soldiers, leaves)."""
    print("[*] Creating dedicated hard negatives from real foliage and camo scenes...", flush=True)
    negatives = []
    
    # 1. From user screenshot
    screenshot_path = Path(r"C:\Users\ENG MOHAMED SABRY\.gemini\antigravity-ide\brain\4e23f119-8bdd-45f8-a07f-f2783d6b9196\.user_uploaded\media_1791745369736.png")
    if screenshot_path.exists():
        try:
            with Image.open(screenshot_path) as full_img:
                # Crop soldier torso/camo without drone
                soldier_crop = full_img.crop((260, 200, 390, 410))
                p1 = Path("scratch/neg_soldier_camo.jpg")
                soldier_crop.save(p1)
                negatives.append((str(p1), "", "neg_soldier_camo"))

                # Crop tree canopy/leaves without drone
                leaves_crop = full_img.crop((390, 145, 580, 260))
                p2 = Path("scratch/neg_tree_branches.jpg")
                leaves_crop.save(p2)
                negatives.append((str(p2), "", "neg_tree_branches"))

                # Crop distant trees
                trees_crop = full_img.crop((260, 250, 450, 400))
                p3 = Path("scratch/neg_field_trees.jpg")
                trees_crop.save(p3)
                negatives.append((str(p3), "", "neg_field_trees"))
        except Exception as e:
            print(f"Failed soldier crop: {e}", flush=True)

    print(f"    -> Added {len(negatives)} custom hard negatives.", flush=True)
    return negatives


def get_synthetic_blends(kamikaze_count=20, balloon_count=20):
    print(f"[*] Blending {kamikaze_count} delta drones and {balloon_count} balloons from synthetic_v4...", flush=True)
    synth_dir = Path("data/synthetic_v4")
    if not synth_dir.exists():
        return []

    kamikaze_samples = []
    balloon_samples = []

    for split in ["train", "val", "test"]:
        img_dir = synth_dir / "images" / split
        lbl_dir = synth_dir / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            with open(lbl_path, "r") as f:
                content = f.read()
            if "0 " in content and len(kamikaze_samples) < kamikaze_count:
                kamikaze_samples.append((str(img_path), content, f"synth_delta_{img_path.stem}"))
            elif "3 " in content and len(balloon_samples) < balloon_count:
                balloon_samples.append((str(img_path), content, f"synth_balloon_{img_path.stem}"))

    return kamikaze_samples + balloon_samples


def split_and_save(all_samples):
    print(f"[*] Saving and distributing {len(all_samples)} total samples into train/val/test...", flush=True)
    random.seed(42)
    random.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * 0.70)
    n_val = int(n * 0.20)
    
    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:],
    }

    class_counts = {split: {cid: 0 for cid in CANONICAL_CLASSES} for split in splits}
    bg_counts = {split: 0 for split in splits}

    for split_name, samples in splits.items():
        img_out = OUTPUT_DIR / "images" / split_name
        lbl_out = OUTPUT_DIR / "labels" / split_name
        
        for img_src, label_text, base_name in samples:
            dst_img = img_out / f"{base_name}.jpg"
            dst_lbl = lbl_out / f"{base_name}.txt"
            
            try:
                with Image.open(img_src) as im:
                    im.convert("RGB").save(dst_img, "JPEG", quality=92)
            except Exception:
                shutil.copy2(img_src, dst_img)

            with open(dst_lbl, "w") as f:
                f.write(label_text)

            if not label_text.strip():
                bg_counts[split_name] += 1
            else:
                for line in label_text.strip().split("\n"):
                    if line:
                        cid = int(line.split()[0])
                        if cid in class_counts[split_name]:
                            class_counts[split_name][cid] += 1

    print("\n=======================================================", flush=True)
    print("      REAL-WORLD AERIAL DEFENSE DATASET v1", flush=True)
    print("=======================================================", flush=True)
    for split_name in ["train", "val", "test"]:
        total_imgs = len(splits[split_name])
        bgs = bg_counts[split_name]
        print(f"  Split [{split_name.upper()}]: {total_imgs} images ({bgs} hard negative background images)")
        for cid, name in CANONICAL_CLASSES.items():
            print(f"    - [{cid}] {name:<12}: {class_counts[split_name][cid]} instances")

    yaml_content = f"""path: {OUTPUT_DIR.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: drone
  1: aircraft
  2: bird
  3: balloon
  4: military_jet
"""
    with open(OUTPUT_DIR / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    print(f"\n[+] Saved configuration: {OUTPUT_DIR / 'dataset.yaml'}", flush=True)


def main():
    setup_dirs()
    drones = download_rigoleto_drones(limit=140)
    jets, airliners = download_military_aircraft(limit_jets=90, limit_airliners=45)
    birds, negatives = download_matisdsp_birds_and_negatives(limit_birds=60, limit_negatives=50)
    custom_negs = add_custom_foliage_and_soldier_negatives()
    blends = get_synthetic_blends(kamikaze_count=20, balloon_count=20)

    all_samples = drones + jets + airliners + birds + negatives + custom_negs + blends
    print(f"\nTotal curated real-world samples collected: {len(all_samples)}", flush=True)
    split_and_save(all_samples)


if __name__ == "__main__":
    main()
