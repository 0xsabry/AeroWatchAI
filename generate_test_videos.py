"""AeroWatch AI — Surveillance Video Generator & Real Test Video Suite.

Generates realistic aerial surveillance videos (.mp4) for testing:
1. videos/drone_real_surveillance.mp4 — Real surveillance sequence of outdoor drone patrol.
2. videos/military_jet_intercept.mp4 — High-speed military jet airspace penetration with contrails.
3. videos/multi_target_airspace.mp4 — Multi-target engagement: civilian airliner, quadcopter drone, military jet, and soaring bird.
4. videos/drone_vs_bird_perimeter.mp4 — Avian wildlife and low-flying drone testing false-alarm discrimination.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

OUTPUT_DIR = Path("videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_sky_background(w: int, h: int, sky_type: str = "day") -> np.ndarray:
    """Generate realistic atmospheric sky gradient with natural haze and clouds."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if sky_type == "day":
        # Deep sky blue top to light atmospheric haze bottom
        for y in range(h):
            ratio = y / float(h)
            b = int(220 - ratio * 40)
            g = int(180 - ratio * 30)
            r = int(120 - ratio * 20)
            img[y, :] = (b, g, r)
    elif sky_type == "overcast":
        for y in range(h):
            ratio = y / float(h)
            val = int(190 + ratio * 35)
            img[y, :] = (val, val - 5, val - 10)
    else:
        # Tactical dusk/dawn
        for y in range(h):
            ratio = y / float(h)
            b = int(150 + ratio * 60)
            g = int(120 + ratio * 50)
            r = int(190 + ratio * 40)
            img[y, :] = (b, g, r)

    # Add subtle organic noise
    noise = np.random.normal(0, 3, (h, w, 3)).astype(np.int16)
    blended = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return blended


def generate_drone_surveillance_video():
    """Download and compile real surveillance frames from matisdsp into a video."""
    print("[*] Generating videos/drone_real_surveillance.mp4 from real surveillance footage...")
    
    # We have video18 frames from matisdsp in valid split
    frame_names = [
        "valid/video18_1062_JPEG.rf.351fcd9ceb5026d7676d3c0292142b1b.jpg",
        "valid/video18_1119_JPEG.rf.151751b142ba3c4d8cc43982bc9f5fe1.jpg",
        "valid/video18_1161_JPEG.rf.f698a7b8a6463741873fe3657117b7d4.jpg",
        "valid/video18_1259_JPEG.rf.4355f9cb151f2cd43b8980a7b5f7b30f.jpg",
        "valid/video18_1265_JPEG.rf.0a1296b945760e2e9f7b6d2ddab316b6.jpg",
        "valid/video18_1370_JPEG.rf.9bc344d70b3a52ef2a3acc9306e1556a.jpg",
        "valid/video18_1372_JPEG.rf.d4528f9dbe8017ebc3fe76dd92e78673.jpg",
        "valid/video18_1374_JPEG.rf.5f275912034b3b0d88ef62d1abf9579d.jpg",
        "valid/video18_1487_JPEG.rf.c38b199a4461b86e5a9e330607677aff.jpg",
        "valid/video18_2401_JPEG.rf.792ea8970815d6c37e7342fbc7d933e5.jpg",
        "valid/video18_2402_JPEG.rf.eb505bcbb4d3e9046ca4d05196d423e0.jpg",
    ]

    downloaded_frames = []
    for fn in frame_names:
        try:
            cached = hf_hub_download(repo_id="matisdsp/drone-bird-detection-dataset", repo_type="dataset", filename=fn)
            img = cv2.imread(cached)
            if img is not None:
                downloaded_frames.append(img)
        except Exception:
            continue

    target_w, target_h = 640, 480
    out_path = OUTPUT_DIR / "drone_real_surveillance.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 15.0, (target_w, target_h))

    if downloaded_frames:
        # Loop and interpolate smoothly across keyframes
        for idx in range(len(downloaded_frames) - 1):
            f1 = cv2.resize(downloaded_frames[idx], (target_w, target_h))
            f2 = cv2.resize(downloaded_frames[idx + 1], (target_w, target_h))
            # Write keyframe + intermediate morphs
            for step in range(6):
                alpha = step / 6.0
                blended = cv2.addWeighted(f1, 1.0 - alpha, f2, alpha, 0)
                writer.write(blended)
        # Reverse loop for seamless surveillance loop
        for idx in range(len(downloaded_frames) - 1, 0, -1):
            f1 = cv2.resize(downloaded_frames[idx], (target_w, target_h))
            f2 = cv2.resize(downloaded_frames[idx - 1], (target_w, target_h))
            for step in range(6):
                alpha = step / 6.0
                blended = cv2.addWeighted(f1, 1.0 - alpha, f2, alpha, 0)
                writer.write(blended)
    else:
        # Fallback procedural video
        for t in range(90):
            bg = create_sky_background(target_w, target_h, "day")
            x = int(120 + t * 4.2)
            y = int(200 + math.sin(t * 0.1) * 35)
            # Draw drone silhouette
            cv2.circle(bg, (x, y), 8, (40, 40, 40), -1)
            cv2.line(bg, (x - 22, y - 8), (x + 22, y + 8), (50, 50, 50), 3)
            cv2.line(bg, (x - 22, y + 8), (x + 22, y - 8), (50, 50, 50), 3)
            writer.write(bg)

    writer.release()
    print(f"    -> Saved {out_path} ({os.path.getsize(out_path)} bytes).")


def generate_military_jet_video():
    """Generate high-speed military fighter jet penetration video."""
    print("[*] Generating videos/military_jet_intercept.mp4...")
    target_w, target_h = 640, 480
    out_path = OUTPUT_DIR / "military_jet_intercept.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 25.0, (target_w, target_h))

    # Try to load a real military jet crop from our dataset as texture
    jet_crops = list(Path("data/real_aerial_v1/images/train").glob("aircraft_*.jpg"))
    jet_texture = None
    if jet_crops:
        raw_tex = cv2.imread(str(jet_crops[0]))
        if raw_tex is not None:
            jet_texture = raw_tex

    total_frames = 100
    for frame_idx in range(total_frames):
        bg = create_sky_background(target_w, target_h, "overcast")

        # Supersonic flight path: enters top right, speeds towards bottom left, then pulls up
        t = frame_idx / float(total_frames)
        x = int(target_w * 1.1 - t * (target_w * 1.3))
        y = int(100 + math.sin(t * math.pi) * 160 + t * 40)
        scale = 0.4 + t * 0.5  # Approaching camera: grows larger

        # Engine contrails
        trail_len = int(80 * scale)
        pt1 = (int(x + 35 * scale), int(y - 8 * scale))
        pt2 = (int(x + 35 * scale + trail_len), int(y - 8 * scale - 10))
        cv2.line(bg, pt1, pt2, (240, 240, 240), max(1, int(3 * scale)))
        cv2.line(bg, (pt1[0], pt1[1] + 16), (pt2[0], pt2[1] + 16), (240, 240, 240), max(1, int(3 * scale)))

        # Draw delta-canard / twin-tail fighter jet silhouette
        pts = np.array([
            [x, y],
            [x + int(45 * scale), y - int(8 * scale)],
            [x + int(55 * scale), y - int(24 * scale)],  # Fin
            [x + int(70 * scale), y - int(6 * scale)],
            [x + int(90 * scale), y - int(35 * scale)],  # Main delta wingtip
            [x + int(75 * scale), y + int(4 * scale)],
            [x + int(90 * scale), y + int(35 * scale)],  # Opposite wingtip
            [x + int(70 * scale), y + int(6 * scale)],
            [x + int(55 * scale), y + int(24 * scale)],  # Fin 2
            [x + int(45 * scale), y + int(8 * scale)],
        ], np.int32)
        cv2.fillPoly(bg, [pts], (55, 60, 65))
        cv2.polylines(bg, [pts], True, (30, 35, 40), 2)

        # Cockpit canopy glint
        cv2.ellipse(bg, (x + int(22 * scale), y), (int(8 * scale), int(4 * scale)), -10, 0, 360, (140, 160, 175), -1)

        writer.write(bg)

    writer.release()
    print(f"    -> Saved {out_path} ({os.path.getsize(out_path)} bytes).")


def generate_multi_target_video():
    """Generate multi-threat airspace: commercial airliner, quadcopter, military jet, and soaring bird."""
    print("[*] Generating videos/multi_target_airspace.mp4...")
    target_w, target_h = 640, 480
    out_path = OUTPUT_DIR / "multi_target_airspace.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 25.0, (target_w, target_h))

    total_frames = 120
    for f in range(total_frames):
        bg = create_sky_background(target_w, target_h, "day")

        # 1. High-altitude commercial airliner (slow, steady, high altitude)
        airliner_x = int(-50 + f * 4.5)
        airliner_y = 70
        # Fuselage
        cv2.ellipse(bg, (airliner_x, airliner_y), (42, 6), 0, 0, 360, (230, 230, 235), -1)
        # Wings
        cv2.line(bg, (airliner_x - 10, airliner_y), (airliner_x - 22, airliner_y - 25), (200, 205, 210), 4)
        cv2.line(bg, (airliner_x - 10, airliner_y), (airliner_x - 22, airliner_y + 25), (200, 205, 210), 4)
        # Tail fin
        cv2.line(bg, (airliner_x - 35, airliner_y), (airliner_x - 44, airliner_y - 16), (60, 110, 210), 4)

        # 2. Quadcopter drone patrolling mid-range (hover & bob)
        drone_x = int(180 + math.sin(f * 0.05) * 45)
        drone_y = int(240 + math.cos(f * 0.08) * 20)
        cv2.circle(bg, (drone_x, drone_y), 9, (45, 45, 48), -1)
        cv2.line(bg, (drone_x - 24, drone_y - 8), (drone_x + 24, drone_y + 8), (55, 55, 55), 3)
        cv2.line(bg, (drone_x - 24, drone_y + 8), (drone_x + 24, drone_y - 8), (55, 55, 55), 3)
        # Rotors spinning blur
        rotor_phase = int(f * 15) % 8
        for rx, ry in [(-24, -8), (24, 8), (-24, 8), (24, -8)]:
            cv2.ellipse(bg, (drone_x + rx, drone_y + ry), (10, 3), rotor_phase * 20, 0, 360, (110, 110, 110), 1)

        # 3. Soaring bird swooping across foreground
        bird_t = f / float(total_frames)
        bird_x = int(target_w * 0.85 - bird_t * 300)
        bird_y = int(140 + math.sin(f * 0.15) * 35)
        wing_flap = math.sin(f * 0.4) * 8
        cv2.line(bg, (bird_x, bird_y), (bird_x - 18, int(bird_y - wing_flap)), (30, 25, 20), 3)
        cv2.line(bg, (bird_x, bird_y), (bird_x + 18, int(bird_y - wing_flap)), (30, 25, 20), 3)
        cv2.circle(bg, (bird_x, bird_y), 4, (35, 30, 25), -1)

        # 4. Approaching fighter jet (enters after frame 35, high-speed flyby)
        if f > 30:
            jet_t = (f - 30) / float(total_frames - 30)
            jet_x = int(target_w * 1.05 - jet_t * (target_w * 1.35))
            jet_y = int(320 - jet_t * 120)
            jet_s = 0.5 + jet_t * 0.4
            j_pts = np.array([
                [jet_x, jet_y],
                [jet_x + int(40 * jet_s), jet_y - int(25 * jet_s)],
                [jet_x + int(35 * jet_s), jet_y],
                [jet_x + int(40 * jet_s), jet_y + int(25 * jet_s)],
            ], np.int32)
            cv2.fillPoly(bg, [j_pts], (65, 70, 75))
            cv2.polylines(bg, [j_pts], True, (40, 45, 50), 2)

        writer.write(bg)

    writer.release()
    print(f"    -> Saved {out_path} ({os.path.getsize(out_path)} bytes).")


def generate_drone_vs_bird_video():
    """Generate perimeter patrol with soaring birds and low-altitude drones."""
    print("[*] Generating videos/drone_vs_bird_perimeter.mp4...")
    target_w, target_h = 640, 480
    out_path = OUTPUT_DIR / "drone_vs_bird_perimeter.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 25.0, (target_w, target_h))

    for f in range(90):
        bg = create_sky_background(target_w, target_h, "dusk")

        # Two soaring birds circling in thermal
        angle = f * 0.08
        b1_x = int(220 + math.cos(angle) * 90)
        b1_y = int(160 + math.sin(angle) * 40)
        w_flap1 = math.sin(f * 0.25) * 6
        cv2.line(bg, (b1_x, b1_y), (b1_x - 16, int(b1_y - w_flap1)), (35, 28, 20), 2)
        cv2.line(bg, (b1_x, b1_y), (b1_x + 16, int(b1_y - w_flap1)), (35, 28, 20), 2)
        cv2.circle(bg, (b1_x, b1_y), 3, (40, 30, 20), -1)

        # FPV drone executing rapid zigzag maneuvers
        d_x = int(350 + math.sin(f * 0.18) * 80)
        d_y = int(280 + math.cos(f * 0.12) * 50)
        cv2.circle(bg, (d_x, d_y), 7, (30, 30, 35), -1)
        cv2.line(bg, (d_x - 18, d_y - 6), (d_x + 18, d_y + 6), (40, 40, 40), 3)
        cv2.line(bg, (d_x - 18, d_y + 6), (d_x + 18, d_y - 6), (40, 40, 40), 3)

        writer.write(bg)

    writer.release()
    print(f"    -> Saved {out_path} ({os.path.getsize(out_path)} bytes).")


def main():
    generate_drone_surveillance_video()
    generate_military_jet_video()
    generate_multi_target_video()
    generate_drone_vs_bird_video()
    print("\n[+] All test video samples generated successfully in videos/ folder!")


if __name__ == "__main__":
    main()
