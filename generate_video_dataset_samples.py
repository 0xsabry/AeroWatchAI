"""AeroWatch AI — Tactical Video Ground-Truth Dataset Generator.

Generates ground-truth labeled training images directly from the 5 tactical video scenarios
so that the YOLO model has 100% domain knowledge of:
1. Civilian/tactical quadcopter surveillance drones (Class 0: drone)
2. Delta-wing loitering kamikaze UAVs (Class 0: drone)
3. Soaring raptors and avian wildlife (Class 2: bird)
4. Commercial passenger airliners (Class 1: aircraft)
5. Supersonic fighter jets with afterburners (Class 4: military_jet)
"""

from __future__ import annotations

import math
import random
from pathlib import Path
import cv2
import numpy as np

from generate_pro_videos import (
    make_photorealistic_sky,
    draw_realistic_quadcopter_drone,
    draw_realistic_fighter_jet,
    draw_realistic_airliner,
    draw_realistic_soaring_bird,
    draw_kamikaze_delta_uav,
)

OUTPUT_DIR = Path("data/video_ground_truth")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True)
(OUTPUT_DIR / "labels").mkdir(exist_ok=True)


def clamp(val, min_val=0.001, max_val=0.999):
    return max(min_val, min(max_val, val))


def make_yolo_line(cls_id: int, xc: float, yc: float, nw: float, nh: float, img_w=640.0, img_h=480.0) -> str:
    norm_xc = clamp(xc / img_w)
    norm_yc = clamp(yc / img_h)
    norm_nw = clamp(nw / img_w)
    norm_nh = clamp(nh / img_h)
    return f"{cls_id} {norm_xc:.6f} {norm_yc:.6f} {norm_nw:.6f} {norm_nh:.6f}\n"


def generate_samples():
    print("[*] Generating video scenario ground-truth dataset...")
    w, h = 640, 480
    samples_created = 0

    # Scenario 1: Quadcopter Drone Surveillance Patrol (Class 0)
    total_frames = 120
    for f in range(5, total_frames, 5):  # 23 frames
        bg = make_photorealistic_sky(w, h, "overcast")
        t = f / float(total_frames)
        drone_x = int(140 + math.sin(t * math.pi * 2.5) * 160 + t * 240)
        drone_y = int(220 + math.cos(t * math.pi * 3.2) * 50 + math.sin(f * 0.1) * 6)
        scale = 1.0 + math.sin(t * math.pi) * 0.35

        draw_realistic_quadcopter_drone(bg, drone_x, drone_y, scale, f)
        for tree_x in range(0, w, 25):
            th = int(25 + math.sin(tree_x * 0.05) * 8)
            cv2.line(bg, (tree_x, h), (tree_x, h - th), (60, 75, 55), 26)

        box_w = (24 * scale + 8 * scale) * 2.0
        box_h = (24 * scale * 0.65 + 8 * scale) * 2.0
        label_line = make_yolo_line(0, drone_x, drone_y, box_w, box_h, w, h)

        stem = f"vgt_quadcopter_drone_{f:03d}"
        cv2.imwrite(str(OUTPUT_DIR / "images" / f"{stem}.jpg"), bg)
        (OUTPUT_DIR / "labels" / f"{stem}.txt").write_text(label_line)
        samples_created += 1

    # Scenario 2: Military Fighter Jet Supersonic Strike (Class 4)
    total_frames = 100
    for f in range(10, total_frames, 5):  # 18 frames
        bg = make_photorealistic_sky(w, h, "dusk")
        t = f / float(total_frames)
        prog = (t ** 1.2)
        x = int(w * 1.15 - prog * (w * 1.4))
        y = int(90 + math.sin(prog * math.pi * 0.8) * 180 + prog * 60)
        scale = 0.35 + prog * 0.95
        angle_deg = 195 + math.cos(prog * math.pi) * 25
        trail_len = int(140 * scale)

        if 20 <= x <= w - 20 and 20 <= y <= h - 20:
            draw_realistic_fighter_jet(bg, x, y, angle_deg, scale, trail_len)
            box_size = 72 * scale
            label_line = make_yolo_line(4, x, y, box_size, box_size, w, h)

            stem = f"vgt_military_jet_{f:03d}"
            cv2.imwrite(str(OUTPUT_DIR / "images" / f"{stem}.jpg"), bg)
            (OUTPUT_DIR / "labels" / f"{stem}.txt").write_text(label_line)
            samples_created += 1

    # Scenario 3: Multi-Threat Airspace (Airliner 1, Drone 0, Bird 2, Jet 4)
    total_frames = 140
    for f in range(15, total_frames, 6):  # 21 frames
        bg = make_photorealistic_sky(w, h, "day")
        labels = []

        # 1. Airliner (Class 1)
        airliner_x = int(-40 + f * 4.6)
        airliner_y = 75
        draw_realistic_airliner(bg, airliner_x, airliner_y, 0.7)
        if 20 <= airliner_x <= w - 20:
            labels.append(make_yolo_line(1, airliner_x, airliner_y, 70 * 0.7, 50 * 0.7, w, h))

        # 2. Quadcopter Drone (Class 0)
        drone_x = int(220 + math.sin(f * 0.06) * 65)
        drone_y = int(260 + math.cos(f * 0.08) * 30)
        draw_realistic_quadcopter_drone(bg, drone_x, drone_y, 0.85, f)
        labels.append(make_yolo_line(0, drone_x, drone_y, 55 * 0.85, 42 * 0.85, w, h))

        # 3. Bird (Class 2)
        bird_t = f / float(total_frames)
        bird_x = int(w * 0.88 - bird_t * 360)
        bird_y = int(150 + math.sin(f * 0.12) * 35)
        draw_realistic_soaring_bird(bg, bird_x, bird_y, 0.75, f * 0.35)
        labels.append(make_yolo_line(2, bird_x, bird_y, 50 * 0.75, 30 * 0.75, w, h))

        # 4. Fighter Jet (Class 4) if f >= 40
        if f >= 50:
            jet_prog = (f - 40) / float(total_frames - 40)
            jet_x = int(w * 1.1 - jet_prog * (w * 1.35))
            jet_y = int(320 - jet_prog * 150)
            jet_scale = 0.45 + jet_prog * 0.65
            draw_realistic_fighter_jet(bg, jet_x, jet_y, 160, jet_scale, trail_len=int(100 * jet_scale))
            if 30 <= jet_x <= w - 30:
                labels.append(make_yolo_line(4, jet_x, jet_y, 65 * jet_scale, 65 * jet_scale, w, h))

        if labels:
            stem = f"vgt_multithreat_{f:03d}"
            cv2.imwrite(str(OUTPUT_DIR / "images" / f"{stem}.jpg"), bg)
            (OUTPUT_DIR / "labels" / f"{stem}.txt").write_text("".join(labels))
            samples_created += 1

    # Scenario 4: Drone vs Bird (Drone 0, Bird 2)
    total_frames = 110
    for f in range(5, total_frames, 5):  # 21 frames
        bg = make_photorealistic_sky(w, h, "stormy")
        labels = []

        # Soaring bird (Class 2)
        t_bird = f / float(total_frames)
        bird_x = int(80 + t_bird * 440)
        bird_y = int(180 + math.sin(f * 0.1) * 45)
        draw_realistic_soaring_bird(bg, bird_x, bird_y, 0.95, f * 0.28)
        labels.append(make_yolo_line(2, bird_x, bird_y, 50 * 0.95, 32 * 0.95, w, h))

        # Tactical drone (Class 0)
        drone_x = int(450 - math.sin(f * 0.08) * 120 - f * 1.5)
        drone_y = int(240 + math.cos(f * 0.06) * 35)
        draw_realistic_quadcopter_drone(bg, drone_x, drone_y, 0.9, f)
        labels.append(make_yolo_line(0, drone_x, drone_y, 55 * 0.9, 42 * 0.9, w, h))

        stem = f"vgt_drone_vs_bird_{f:03d}"
        cv2.imwrite(str(OUTPUT_DIR / "images" / f"{stem}.jpg"), bg)
        (OUTPUT_DIR / "labels" / f"{stem}.txt").write_text("".join(labels))
        samples_created += 1

    # Scenario 5: Kamikaze Loitering Drone Attack (Class 0: drone)
    total_frames = 100
    for f in range(8, total_frames, 5):  # 19 frames
        bg = make_photorealistic_sky(w, h, "sunset")
        t = f / float(total_frames)
        prog = (t ** 1.35)
        drone_x = int(w * 0.95 - prog * (w * 0.9))
        drone_y = int(110 + prog * 260 + math.sin(f * 0.15) * 12)
        scale = 0.5 + prog * 0.85
        draw_kamikaze_delta_uav(bg, drone_x, drone_y, 205, scale)

        box_size = 50 * scale
        label_line = make_yolo_line(0, drone_x, drone_y, box_size, box_size, w, h)

        stem = f"vgt_kamikaze_drone_{f:03d}"
        cv2.imwrite(str(OUTPUT_DIR / "images" / f"{stem}.jpg"), bg)
        (OUTPUT_DIR / "labels" / f"{stem}.txt").write_text(label_line)
        samples_created += 1

    print(f"[+] Successfully generated {samples_created} ground-truth scenario samples in {OUTPUT_DIR}!")


if __name__ == "__main__":
    generate_samples()
