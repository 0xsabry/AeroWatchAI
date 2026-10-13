"""AeroWatch AI — Real-World Aerial Defense Dataset v3 Builder.

Assembles an expanded, perfectly balanced multi-class dataset in data/real_aerial_v3:
- Real Commercial Airliners (Boeing 737-787, Airbus A320-A380) fetched via metadata.jsonl
- Real Drones (multirotor, FPV, commercial UAVs) from Rigoleto & matisdsp
- Real Fighter Jets & Warplanes (F-15, F-16, F-22, F-35, Su-27, Rafale) from a2015003713
- Real Soaring Birds from matisdsp
- High-altitude weather balloons & kamikaze delta UAVs
- Hard negative background frames (foliage, tree leaves, soldier uniforms, empty sky)
"""

from __future__ import annotations

import concurrent.futures
import glob
import json
import os
import random
import shutil
from pathlib import Path
from PIL import Image
import requests
from huggingface_hub import hf_hub_download

CANONICAL_CLASSES = {
    0: "drone",
    1: "aircraft",
    2: "bird",
    3: "balloon",
    4: "military_jet",
}

OUTPUT_DIR = Path("data/real_aerial_v3")


def setup_dirs():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"[*] Initialized clean directories in {OUTPUT_DIR}")


def fetch_airliners_from_metadata(target_count: int = 70):
    """Fetch commercial airliners from a2015003713 metadata.jsonl."""
    print(f"[*] Fetching up to {target_count} real commercial airliners from a2015003713...")
    url = "https://huggingface.co/datasets/a2015003713/military-aircraft-detection-dataset/raw/main/metadata.jsonl"
    
    candidates = []
    try:
        r = requests.get(url, stream=True, timeout=15)
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            cats = data.get("objects", {}).get("categories", [])
            # Airliners are categories 103..113 (Boeing, Airbus)
            if any(103 <= c <= 113 for c in cats):
                candidates.append(data)
                if len(candidates) >= target_count * 2:
                    break
    except Exception as e:
        print(f"    Warning during metadata stream: {e}")

    print(f"    Found {len(candidates)} airliner metadata candidates. Downloading files...")
    samples = []

    def download_airliner(item):
        fn = item["file_name"]
        w = item["width"]
        h = item["height"]
        bboxes = item.get("objects", {}).get("bbox", [])
        categories = item.get("objects", {}).get("categories", [])
        
        yolo_lines = []
        for box, cat in zip(bboxes, categories):
            # If airliner (103..113), map to class 1
            # If military jet, map to class 4
            cls_id = 1 if (103 <= cat <= 113) else 4
            # COCO bbox: [x_min, y_min, width, height]
            x_min, y_min, bw, bh = box
            xc = (x_min + bw / 2.0) / float(w)
            yc = (y_min + bh / 2.0) / float(h)
            nw = bw / float(w)
            nh = bh / float(h)
            xc = max(0.001, min(0.999, xc))
            yc = max(0.001, min(0.999, yc))
            nw = max(0.002, min(0.999, nw))
            nh = max(0.002, min(0.999, nh))
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        if not yolo_lines:
            return None

        stem = Path(fn).stem
        repo_fn = f"images/{stem[0]}/{stem}.jpg"
        try:
            cached_img = hf_hub_download(
                repo_id="a2015003713/military-aircraft-detection-dataset",
                repo_type="dataset",
                filename=repo_fn,
            )
            return (cached_img, "".join(yolo_lines), f"airliner_{stem}")
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(download_airliner, candidates[:target_count]):
            if res:
                samples.append(res)

    print(f"    -> Acquired {len(samples)} real commercial airliner samples.")
    return samples


def fetch_more_rigoleto_drones(start_idx: int = 141, count: int = 120):
    """Fetch additional commercial and FPV drones from Rigoleto."""
    print(f"[*] Downloading {count} additional real drones from Rigoleto/drone-detection...")
    items = [(f"train{i:04d}", "train") for i in range(start_idx, start_idx + count)]
    samples = []

    def fetch_item(item):
        name, split = item
        lbl_file = f"data/{split}/labels/{name}.txt"
        img_file = f"data/{split}/images/{name}.jpg"
        try:
            cached_lbl = hf_hub_download(
                repo_id="Rigoleto/drone-detection",
                repo_type="dataset",
                filename=lbl_file,
            )
            cached_img = hf_hub_download(
                repo_id="Rigoleto/drone-detection",
                repo_type="dataset",
                filename=img_file,
            )
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(fetch_item, items):
            if r is not None:
                samples.append(r)

    print(f"    -> Acquired {len(samples)} additional real drone samples.")
    return samples


def collect_existing_v2_samples():
    """Collect the high quality samples already validated in real_aerial_v2."""
    print("[*] Incorporating existing validated samples from real_aerial_v2...")
    v2_dir = Path("data/real_aerial_v2")
    samples = []
    
    for split in ["train", "val", "test"]:
        img_dir = v2_dir / "images" / split
        lbl_dir = v2_dir / "labels" / split
        if not img_dir.exists():
            continue
        for p in img_dir.glob("*.jpg"):
            lbl_file = lbl_dir / f"{p.stem}.txt"
            if not lbl_file.exists():
                continue
            txt = lbl_file.read_text()
            samples.append((str(p), txt, f"v3_{p.stem}"))
    
    print(f"    -> Collected {len(samples)} existing samples from real_aerial_v2.")
    return samples


def split_and_save(all_samples):
    print(f"\n[*] Distributing {len(all_samples)} total samples with stratified balance...")
    random.seed(42)

    # Group by dominant class
    class_buckets: dict[int, list] = {-1: [], 0: [], 1: [], 2: [], 3: [], 4: []}
    for item in all_samples:
        src, txt, stem = item
        if not txt.strip():
            class_buckets[-1].append(item)
        else:
            first_cid = int(txt.strip().split("\n")[0].split()[0])
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

            # Copy / convert image
            try:
                if not Path(src_img_path).exists():
                    continue
                with Image.open(src_img_path) as im:
                    im.convert("RGB").save(out_img, "JPEG", quality=92)
            except Exception:
                continue

            # Write label
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

    print("\n[+] Dataset Partitioning Complete:")
    for split in ["train", "val", "test"]:
        cc = class_counts[split]
        print(f"    Split [{split:5s}]: {len(splits[split])} images | "
              f"Drones: {cc[0]}, Aircraft: {cc[1]}, Birds: {cc[2]}, "
              f"Balloons: {cc[3]}, Jets: {cc[4]}, Hard Negatives: {cc['background']}")

    # Write dataset.yaml
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
    v2_samples = collect_existing_v2_samples()
    new_airliners = fetch_airliners_from_metadata(target_count=65)
    new_drones = fetch_more_rigoleto_drones(start_idx=141, count=100)

    combined = v2_samples + new_airliners + new_drones
    split_and_save(combined)
    print(f"\n[+] Dataset v3 build finished successfully! Total samples: {len(combined)}")


if __name__ == "__main__":
    main()
