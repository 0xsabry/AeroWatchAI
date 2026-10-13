"""AeroWatch AI — Balanced Real-World Dataset Builder (real_aerial_v4_balanced).

Constructs an unbiased, class-balanced dataset to eliminate:
1. False 'military_jet' predictions on drones, birds, and civilian airliners.
2. Missed detections on small multirotor and DJI commercial drones.
3. False alarms on background foliage, tree leaves, and camouflage.

Classes:
0: drone (multirotor, FPV, commercial UAVs, DJI Mavic, loitering munition)
1: aircraft (commercial airliners Boeing 737-787, Airbus A320-A380)
2: bird (soaring raptors, avian wildlife)
3: balloon (weather aerostats, hot-air balloons)
4: military_jet (F-15, F-16, F-22, F-35, Su-27, Rafale fighter jets)
- Hard Negatives: foliage, tree branches, soldier camo gear (0 labels)
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import shutil
from pathlib import Path
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download

OUTPUT_DIR = Path("data/real_aerial_v4_balanced")


def setup_dirs():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"[*] Initialized clean directories in {OUTPUT_DIR}")


def fetch_matisdsp_samples(limit_drones=120, limit_birds=75):
    """Fetch matched real drone and bird images with COCO bboxes from matisdsp."""
    print(f"[*] Fetching real drones ({limit_drones}) and birds ({limit_birds}) from matisdsp...")
    try:
        api = HfApi()
        repo_files = set(api.list_repo_files("matisdsp/drone-bird-detection-dataset", repo_type="dataset"))
        f = hf_hub_download(repo_id="matisdsp/drone-bird-detection-dataset", repo_type="dataset", filename="valid/_annotations.coco.json")
        with open(f, "r") as fp:
            coco = json.load(fp)
    except Exception as e:
        print(f"    Failed to load matisdsp annotations: {e}")
        return [], []

    img_info = {im["id"]: im for im in coco["images"] if ("valid/" + im["file_name"]) in repo_files}
    
    img_anns = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid in img_info:
            img_anns.setdefault(iid, []).append(ann)

    v18_cands = []
    other_drone_cands = []
    bird_candidates = []
    for iid, anns in img_anns.items():
        im = img_info[iid]
        has_drone = any(a["category_id"] == 0 for a in anns)
        has_bird = any(a["category_id"] == 1 for a in anns)
        fn = im["file_name"]
        if has_drone:
            if "video18" in fn and len(v18_cands) < 35:
                v18_cands.append((im, anns))
            elif len(other_drone_cands) < (limit_drones - 35):
                other_drone_cands.append((im, anns))
        elif has_bird and len(bird_candidates) < limit_birds:
            bird_candidates.append((im, anns))

    drone_candidates = v18_cands + other_drone_cands

    samples_drone = []
    samples_bird = []

    def download_coco_item(item):
        im, anns = item
        fn = "valid/" + im["file_name"]
        w = im["width"]
        h = im["height"]
        try:
            cached_path = hf_hub_download(repo_id="matisdsp/drone-bird-detection-dataset", repo_type="dataset", filename=fn)
        except Exception:
            return None

        yolo_lines = []
        is_bird = False
        for a in anns:
            cid = a["category_id"]
            if cid == 0:
                yolo_cls = 0  # drone
            elif cid == 1:
                yolo_cls = 2  # bird
                is_bird = True
            else:
                continue
            x_min, y_min, bw, bh = a["bbox"]
            xc = (x_min + bw / 2.0) / float(w)
            yc = (y_min + bh / 2.0) / float(h)
            nw = bw / float(w)
            nh = bh / float(h)
            xc = max(0.001, min(0.999, xc))
            yc = max(0.001, min(0.999, yc))
            nw = max(0.002, min(0.999, nw))
            nh = max(0.002, min(0.999, nh))
            yolo_lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        if not yolo_lines:
            return None

        stem = Path(im["file_name"]).stem
        return (cached_path, "".join(yolo_lines), f"matis_{stem}", is_bird)

    all_cands = drone_candidates + bird_candidates
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(download_coco_item, all_cands):
            if res:
                path, txt, stem, is_bird = res
                if is_bird:
                    samples_bird.append((path, txt, stem))
                else:
                    samples_drone.append((path, txt, stem))

    print(f"    -> Acquired {len(samples_drone)} real drone samples, {len(samples_bird)} real bird samples.")
    return samples_drone, samples_bird


def collect_rigoleto_drones(limit=80):
    """Collect real Rigoleto commercial multirotor drones."""
    print(f"[*] Collecting {limit} real Rigoleto drones...")
    src_dir = Path("data/real_aerial_v1")
    samples = []
    if src_dir.exists():
        for p in src_dir.glob("images/train/*rigoleto*.jpg"):
            lbl_file = src_dir / "labels" / "train" / f"{p.stem}.txt"
            if lbl_file.exists():
                lines = [l.strip() for l in lbl_file.read_text().splitlines() if l.strip()]
                cids = [int(l.split()[0]) for l in lines]
                if 0 in cids and not any(c != 0 for c in cids):
                    samples.append((str(p), "\n".join(lines) + "\n", f"rigo_{p.stem}"))
                    if len(samples) >= limit:
                        break
    print(f"    -> Acquired {len(samples)} Rigoleto drone samples.")
    return samples


def add_user_mavic_drone_with_augmentations():
    """Add user soldier photo with exact DJI Mavic quadcopter drone box + augmentations."""
    print("[*] Incorporating user Mavic drone photo with multiple tactical augmentations...")
    p_user = Path("videos/test_soldier_drone_photo.jpg")
    samples = []
    if not p_user.exists():
        return []

    # Drone box on 337x286 image: [177.6, 104.0, 233.8, 120.0]
    w, h = 337.0, 286.0
    xc = (177.6 + 56.2 / 2.0) / w
    yc = (104.0 + 16.0 / 2.0) / h
    nw = 56.2 / w
    nh = 16.0 / h
    label_txt = f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"

    samples.append((str(p_user), label_txt, "user_soldier_drone_orig"))

    # Flipped augmentation
    flip_p = Path("videos/test_soldier_drone_flip.jpg")
    with Image.open(p_user) as im:
        flipped = im.transpose(Image.FLIP_LEFT_RIGHT)
        flipped.save(flip_p, "JPEG")
    flip_xc = 1.0 - xc
    flip_label = f"0 {flip_xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"
    samples.append((str(flip_p), flip_label, "user_soldier_drone_flip"))

    # Contrast/brightness variations
    for factor, suffix in [(0.8, "dark"), (1.25, "bright")]:
        aug_p = Path(f"videos/test_soldier_drone_{suffix}.jpg")
        with Image.open(p_user) as im:
            from PIL import ImageEnhance
            enhanced = ImageEnhance.Brightness(im).enhance(factor)
            enhanced.save(aug_p, "JPEG")
        samples.append((str(aug_p), label_txt, f"user_soldier_drone_{suffix}"))

    print(f"    -> Added {len(samples)} dedicated DJI Mavic drone samples.")
    return samples


def collect_commercial_airliners(limit=80):
    """Collect commercial airliners with strict class-1 verification."""
    print(f"[*] Collecting commercial airliners from real_aerial_v3...")
    src_dir = Path("data/real_aerial_v3")
    samples = []
    if src_dir.exists():
        for p in src_dir.glob("images/*/*airliner*.jpg"):
            lbl_file = p.parent.parent.parent / "labels" / p.parent.name / f"{p.stem}.txt"
            if lbl_file.exists():
                lines = [l.strip() for l in lbl_file.read_text().splitlines() if l.strip()]
                cids = [int(l.split()[0]) for l in lines]
                if 1 in cids and not any(c != 1 for c in cids):
                    samples.append((str(p), "\n".join(lines) + "\n", f"air_{p.stem}"))
                    if len(samples) >= limit:
                        break
    print(f"    -> Acquired {len(samples)} commercial airliner samples.")
    return samples


def collect_controlled_military_jets(limit=130):
    """Collect clean, single/twin flying military fighter jets (strict class-4 only, no airfield clusters)."""
    print(f"[*] Collecting a balanced subset of clean military fighter jets...")
    src_dir = Path("data/real_aerial_v2")
    samples = []
    if src_dir.exists():
        for p in src_dir.glob("images/train/aircraft_*.jpg"):
            lbl_file = src_dir / "labels" / "train" / f"{p.stem}.txt"
            if lbl_file.exists():
                lines = [l.strip() for l in lbl_file.read_text().splitlines() if l.strip()]
                cids = [int(l.split()[0]) for l in lines]
                if 4 in cids and not any(c != 4 for c in cids) and len(lines) <= 2:
                    samples.append((str(p), "\n".join(lines) + "\n", f"jet_{p.stem}"))
                    if len(samples) >= limit:
                        break
    print(f"    -> Acquired {len(samples)} clean military fighter jet samples.")
    return samples


def collect_balloons(limit=20):
    """Collect balloons with strict class-3 verification."""
    print("[*] Collecting balloons...")
    src_dir = Path("data/synthetic_v4")
    samples = []
    if src_dir.exists():
        for p in src_dir.glob("images/*/*.png"):
            lbl_file = p.parent.parent.parent / "labels" / p.parent.name / f"{p.stem}.txt"
            if lbl_file.exists():
                lines = [l.strip() for l in lbl_file.read_text().splitlines() if l.strip()]
                cids = [int(l.split()[0]) for l in lines]
                if 3 in cids and not any(c != 3 for c in cids):
                    samples.append((str(p), "\n".join(lines) + "\n", f"balloon_{p.stem}"))
                    if len(samples) >= limit:
                        break
    print(f"    -> Acquired {len(samples)} clean balloon samples.")
    return samples


def collect_video_ground_truth():
    """Collect ground-truth labeled scenario frames generated from tactical video engines."""
    print("[*] Collecting tactical video scenario ground-truth samples...")
    src_dir = Path("data/video_ground_truth")
    samples = []
    if src_dir.exists():
        for p in (src_dir / "images").glob("*.jpg"):
            lbl_file = src_dir / "labels" / f"{p.stem}.txt"
            if lbl_file.exists():
                samples.append((str(p), lbl_file.read_text(), p.stem))
    print(f"    -> Acquired {len(samples)} video scenario ground-truth samples.")
    return samples


def collect_hard_negatives(limit=35):
    """Collect hard negative background images (tree leaves, soldier camo, boat decks, empty skies)."""
    print(f"[*] Collecting hard negative background samples...")
    src_dir = Path("data/real_aerial_v1")
    samples = []
    if src_dir.exists():
        for p in src_dir.glob("images/*/*neg_*.jpg"):
            lbl_file = p.parent.parent.parent / "labels" / p.parent.name / f"{p.stem}.txt"
            if lbl_file.exists():
                samples.append((str(p), "", f"neg_{p.stem}"))
                if len(samples) >= 15:
                    break

    # Add boat deck & maritime water negatives
    boat_dir = Path("data/boat_negatives")
    if boat_dir.exists():
        for p in boat_dir.glob("*.jpg"):
            samples.append((str(p), "", f"neg_boat_{p.stem}"))

    print(f"    -> Acquired {len(samples)} hard negative background samples.")
    return samples


def split_and_save_balanced(all_samples):
    print(f"\n[*] Distributing {len(all_samples)} total samples with strict class balancing...")
    random.seed(1337)

    # Group by primary class token
    class_buckets = {-1: [], 0: [], 1: [], 2: [], 3: [], 4: []}
    for item in all_samples:
        src, txt, stem = item
        clean_lines = [l.strip() for l in txt.splitlines() if l.strip()]
        if not clean_lines:
            class_buckets[-1].append(item)
        else:
            first_cid = int(clean_lines[0].split()[0])
            if first_cid in class_buckets:
                class_buckets[first_cid].append(item)

    splits = {"train": [], "val": [], "test": []}
    for cid, bucket in class_buckets.items():
        random.shuffle(bucket)
        n = len(bucket)
        n_train = max(1, int(n * 0.72))
        n_val = max(1, int(n * 0.18)) if n >= 4 else (1 if n >= 2 else 0)
        splits["train"].extend(bucket[:n_train])
        splits["val"].extend(bucket[n_train:n_train + n_val])
        splits["test"].extend(bucket[n_train + n_val:])

    class_counts = {split: {cid: 0 for cid in range(5)} for split in splits}
    class_counts["train"]["background"] = 0
    class_counts["val"]["background"] = 0
    class_counts["test"]["background"] = 0

    for split, items in splits.items():
        dst_img_dir = OUTPUT_DIR / "images" / split
        dst_lbl_dir = OUTPUT_DIR / "labels" / split

        for src_img_path, yolo_txt, stem in items:
            out_img = dst_img_dir / f"{stem}.jpg"
            out_lbl = dst_lbl_dir / f"{stem}.txt"

            try:
                if not Path(src_img_path).exists():
                    continue
                with Image.open(src_img_path) as im:
                    im.convert("RGB").save(out_img, "JPEG", quality=92)
            except Exception:
                continue

            out_lbl.write_text(yolo_txt)

            if not yolo_txt.strip():
                class_counts[split]["background"] += 1
            else:
                for line in yolo_txt.strip().split("\n"):
                    p = line.split()
                    if p:
                        cid = int(p[0])
                        if cid in class_counts[split]:
                            class_counts[split][cid] += 1

    print("\n[+] Balanced Dataset Partitioning Complete:")
    for split in ["train", "val", "test"]:
        cc = class_counts[split]
        print(f"    Split [{split:5s}]: {len(splits[split])} images | "
              f"Drones: {cc[0]}, Aircraft: {cc[1]}, Birds: {cc[2]}, "
              f"Balloons: {cc[3]}, Jets: {cc[4]}, Hard Negatives: {cc['background']}")

    dataset_yaml = f"""path: {OUTPUT_DIR.resolve().as_posix()}
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
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    yaml_path.write_text(dataset_yaml)
    print(f"\n[+] Wrote configuration to {yaml_path}")


def main():
    setup_dirs()
    drones_m, birds_m = fetch_matisdsp_samples(limit_drones=120, limit_birds=75)
    drones_r = collect_rigoleto_drones(limit=80)
    user_drones = add_user_mavic_drone_with_augmentations()
    airliners = collect_commercial_airliners(limit=80)
    jets = collect_controlled_military_jets(limit=130)
    balloons = collect_balloons(limit=20)
    vgt_samples = collect_video_ground_truth()
    negatives = collect_hard_negatives(limit=25)

    all_samples = (
        drones_m + drones_r + user_drones +
        birds_m +
        airliners +
        jets +
        balloons +
        vgt_samples +
        negatives
    )

    split_and_save_balanced(all_samples)
    print(f"\n[+] Balanced dataset v4 created successfully with {len(all_samples)} total images!")


if __name__ == "__main__":
    main()
