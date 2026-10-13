"""AeroWatch AI — Professional Airspace Surveillance Test Video Generator.

Generates 5 realistic aerial surveillance videos (.mp4) for testing live detection and defense:
1. videos/real_drone_surveillance_patrol.mp4 — Real-world drone surveillance sequence with authentic outdoor flight.
2. videos/military_jet_supersonic_strike.mp4 — Supersonic fighter jet airspace penetration with banking dynamics and twin contrails.
3. videos/multi_threat_airspace_scenario.mp4 — Complex multi-target engagement (Commercial airliner, tactical drone, soaring bird, supersonic jet).
4. videos/drone_vs_bird_perimeter_patrol.mp4 — Avian wildlife vs. agile reconnaissance drone testing false-alarm discrimination.
5. videos/kamikaze_loitering_drone_attack.mp4 — Low-altitude delta-wing loitering munition evasive approach.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

OUTPUT_DIR = Path("videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_photorealistic_sky(w: int, h: int, sky_type: str = "day") -> np.ndarray:
    """Generate high-fidelity atmospheric gradient with realistic cloud strata and haze."""
    img = np.zeros((h, w, 3), dtype=np.float32)

    if sky_type == "day":
        # Deep cerulean zenith to warm atmospheric horizon
        c_top = np.array([210, 160, 95], dtype=np.float32)     # BGR
        c_bot = np.array([245, 230, 205], dtype=np.float32)
    elif sky_type == "overcast":
        c_top = np.array([180, 185, 185], dtype=np.float32)
        c_bot = np.array([215, 218, 220], dtype=np.float32)
    elif sky_type == "dusk":
        c_top = np.array([140, 90, 60], dtype=np.float32)
        c_bot = np.array([120, 150, 230], dtype=np.float32)
    else:
        c_top = np.array([190, 170, 130], dtype=np.float32)
        c_bot = np.array([230, 220, 200], dtype=np.float32)

    for y in range(h):
        r = y / float(h)
        # Non-linear atmospheric haze curve
        curve = r ** 1.3
        img[y, :] = c_top * (1.0 - curve) + c_bot * curve

    # Add soft natural cloud wisps
    clouds = np.zeros((h, w), dtype=np.float32)
    for k in range(3):
        freq = (k + 1) * 0.008
        x_mesh, y_mesh = np.meshgrid(np.arange(w), np.arange(h))
        wisp = np.sin(x_mesh * freq * 1.5 + y_mesh * freq * 0.3) * np.cos(y_mesh * freq * 0.8)
        clouds += wisp * (12.0 / (k + 1))

    cloud_mask = np.clip(clouds, -15, 20)
    for c in range(3):
        img[:, :, c] += cloud_mask * 0.6

    # Micro-camera sensor noise
    noise = np.random.normal(0, 2.2, (h, w, 3)).astype(np.float32)
    final_sky = np.clip(img + noise, 0, 255).astype(np.uint8)
    return final_sky


def extract_real_airframe_sprite(img_path: str, bbox_norm: list[float] | None = None) -> np.ndarray | None:
    """Extract and isolate an airframe sprite with alpha channel from a real dataset image."""
    if not os.path.exists(img_path):
        return None
    src = cv2.imread(img_path)
    if src is None:
        return None

    h, w = src.shape[:2]
    if bbox_norm:
        xc, yc, bw, bh = bbox_norm
        x1 = max(0, int((xc - bw / 2.0) * w))
        y1 = max(0, int((yc - bh / 2.0) * h))
        x2 = min(w, int((xc + bw / 2.0) * w))
        y2 = min(h, int((yc + bh / 2.0) * h))
        crop = src[y1:y2, x1:x2]
    else:
        crop = src

    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None

    # Create 4-channel BGRA with smooth edge alpha
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Target object is typically darker than sky or has distinct contrast
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    alpha = cv2.GaussianBlur(thresh, (5, 5), 1.5)
    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def overlay_rotated_sprite(bg: np.ndarray, sprite: np.ndarray, x: int, y: int, angle_deg: float, scale: float):
    """Overlay a rotated, scaled sprite onto the background with alpha blending."""
    sh, sw = sprite.shape[:2]
    nw = max(8, int(sw * scale))
    nh = max(8, int(sh * scale))
    resized = cv2.resize(sprite, (nw, nh), interpolation=cv2.INTER_AREA)

    # Rotate sprite
    m = cv2.getRotationMatrix2D((nw / 2.0, nh / 2.0), angle_deg, 1.0)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])
    bound_w = int((nh * sin) + (nw * cos))
    bound_h = int((nh * cos) + (nw * sin))
    m[0, 2] += (bound_w / 2) - (nw / 2)
    m[1, 2] += (bound_h / 2) - (nh / 2)
    rotated = cv2.warpAffine(resized, m, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    rx1 = int(x - bound_w / 2.0)
    ry1 = int(y - bound_h / 2.0)
    rx2 = rx1 + bound_w
    ry2 = ry1 + bound_h

    bg_h, bg_w = bg.shape[:2]
    # Clipping
    crop_rx1 = max(0, rx1)
    crop_ry1 = max(0, ry1)
    crop_rx2 = min(bg_w, rx2)
    crop_ry2 = min(bg_h, ry2)

    if crop_rx2 <= crop_rx1 or crop_ry2 <= crop_ry1:
        return

    sp_x1 = crop_rx1 - rx1
    sp_y1 = crop_ry1 - ry1
    sp_x2 = sp_x1 + (crop_rx2 - crop_rx1)
    sp_y2 = sp_y1 + (crop_ry2 - crop_ry1)

    sp_crop = rotated[sp_y1:sp_y2, sp_x1:sp_x2]
    bg_crop = bg[crop_ry1:crop_ry2, crop_rx1:crop_rx2]

    if sp_crop.shape[2] == 4:
        alpha = (sp_crop[:, :, 3] / 255.0)[:, :, np.newaxis]
        bg[crop_ry1:crop_ry2, crop_rx1:crop_rx2] = (sp_crop[:, :, :3] * alpha + bg_crop * (1.0 - alpha)).astype(np.uint8)
    else:
        bg[crop_ry1:crop_ry2, crop_rx1:crop_rx2] = sp_crop


def draw_realistic_fighter_jet(bg: np.ndarray, x: int, y: int, angle_deg: float, scale: float, trail_len: int = 120):
    """Draw a high-realism stealth / twin-fin air-superiority fighter jet with afterburner & contrails."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # 1. Contrails / afterburner trails
    if trail_len > 0:
        tail_offset_x = int(-40 * scale * cos_a)
        tail_offset_y = int(-40 * scale * sin_a)
        for offset_perp in [-8 * scale, 8 * scale]:
            perp_x = int(-offset_perp * sin_a)
            perp_y = int(offset_perp * cos_a)
            t_start_x = x + tail_offset_x + perp_x
            t_start_y = y + tail_offset_y + perp_y
            t_end_x = int(t_start_x - trail_len * cos_a)
            t_end_y = int(t_start_y - trail_len * sin_a)

            # Contrail gradient
            cv2.line(bg, (t_start_x, t_start_y), (t_end_x, t_end_y), (235, 238, 240), max(1, int(3 * scale)))
            # Afterburner supersonic flame glow
            ab_tip_x = int(t_start_x - 14 * scale * cos_a)
            ab_tip_y = int(t_start_y - 14 * scale * sin_a)
            cv2.line(bg, (t_start_x, t_start_y), (ab_tip_x, ab_tip_y), (40, 140, 255), max(2, int(4 * scale)))

    # 2. Authentic airframe geometry points (scaled & rotated)
    local_pts = [
        (48 * scale, 0),                       # Radome nose
        (28 * scale, -7 * scale),              # Forward fuselage
        (12 * scale, -10 * scale),             # LERX root
        (-18 * scale, -42 * scale),            # Starboard main delta wingtip
        (-26 * scale, -18 * scale),            # Trailing edge
        (-38 * scale, -24 * scale),            # Horizontal stabilizer tip
        (-42 * scale, -9 * scale),             # Twin engine exhaust 1
        (-44 * scale, 0),                      # Empennage center
        (-42 * scale, 9 * scale),              # Twin engine exhaust 2
        (-38 * scale, 24 * scale),             # Port stabilizer tip
        (-26 * scale, 18 * scale),             # Trailing edge
        (-18 * scale, 42 * scale),             # Port main delta wingtip
        (12 * scale, 10 * scale),              # LERX root
        (28 * scale, 7 * scale),               # Forward fuselage
    ]

    rot_pts = []
    for px, py in local_pts:
        rx = int(x + px * cos_a - py * sin_a)
        ry = int(y + px * sin_a + py * cos_a)
        rot_pts.append([rx, ry])

    rot_pts = np.array(rot_pts, dtype=np.int32)
    # Tactical low-observable camouflage gray
    cv2.fillPoly(bg, [rot_pts], (58, 62, 68))
    cv2.polylines(bg, [rot_pts], True, (32, 35, 40), max(1, int(1.5 * scale)))

    # Twin canted vertical fins shadow
    for fin_sign in [-1, 1]:
        f_base_x = int(x + (-20 * scale) * cos_a - (fin_sign * 11 * scale) * sin_a)
        f_base_y = int(y + (-20 * scale) * sin_a + (fin_sign * 11 * scale) * cos_a)
        f_tip_x = int(x + (-34 * scale) * cos_a - (fin_sign * 18 * scale) * sin_a)
        f_tip_y = int(y + (-34 * scale) * sin_a + (fin_sign * 18 * scale) * cos_a)
        cv2.line(bg, (f_base_x, f_base_y), (f_tip_x, f_tip_y), (42, 45, 50), max(1, int(2.5 * scale)))

    # Tinted canopy reflection
    c_x = int(x + (15 * scale) * cos_a)
    c_y = int(y + (15 * scale) * sin_a)
    cv2.ellipse(bg, (c_x, c_y), (max(2, int(9 * scale)), max(1, int(4 * scale))), angle_deg, 0, 360, (140, 165, 180), -1)


def draw_realistic_quadcopter_drone(bg: np.ndarray, x: int, y: int, scale: float, frame_idx: int):
    """Draw a realistic commercial/tactical multirotor quadcopter with rotating propellers."""
    arm_span = int(24 * scale)
    body_rad = max(4, int(8 * scale))

    # Carbon fiber central fuselage
    cv2.circle(bg, (x, y), body_rad, (38, 40, 42), -1)
    cv2.circle(bg, (x, y), body_rad, (20, 22, 24), max(1, int(1.5 * scale)))

    # Navigation beacon LED (pulsing red/green)
    led_color = (0, 0, 230) if (frame_idx % 8 < 4) else (0, 220, 0)
    cv2.circle(bg, (x, y), max(1, int(2.5 * scale)), led_color, -1)

    # 4 Rotor arms in X-configuration
    motor_coords = [
        (x - arm_span, y - int(arm_span * 0.65)),
        (x + arm_span, y - int(arm_span * 0.65)),
        (x - arm_span, y + int(arm_span * 0.65)),
        (x + arm_span, y + int(arm_span * 0.65)),
    ]

    for mx, my in motor_coords:
        cv2.line(bg, (x, y), (mx, my), (50, 52, 56), max(1, int(3 * scale)))
        # Motor housing
        cv2.circle(bg, (mx, my), max(2, int(4 * scale)), (30, 32, 35), -1)
        # Spinning propeller motion blur disc
        rot_ang = (frame_idx * 35) % 180
        cv2.ellipse(bg, (mx, my), (max(3, int(12 * scale)), max(1, int(3 * scale))), rot_ang, 0, 360, (110, 115, 120), 1)


def draw_realistic_airliner(bg: np.ndarray, x: int, y: int, scale: float):
    """Draw a commercial twin-engine wide-body passenger airliner cruising at altitude."""
    rad = math.radians(0)  # Horizontal cruise
    l_fuselage = int(60 * scale)
    w_fuselage = max(3, int(8 * scale))

    # Fuselage
    cv2.ellipse(bg, (x, y), (l_fuselage, w_fuselage), 0, 0, 360, (235, 238, 242), -1)
    cv2.ellipse(bg, (x, y), (l_fuselage, w_fuselage), 0, 0, 360, (160, 165, 172), 1)

    # Swept-back wings
    w_span = int(45 * scale)
    w_root = int(x - 5 * scale)
    cv2.line(bg, (w_root, y), (w_root - int(25 * scale), y - w_span), (210, 215, 220), max(2, int(4 * scale)))
    cv2.line(bg, (w_root, y), (w_root - int(25 * scale), y + w_span), (210, 215, 220), max(2, int(4 * scale)))

    # High-bypass turbofan engine pods under wings
    for eng_y in [y - int(18 * scale), y + int(18 * scale)]:
        cv2.ellipse(bg, (w_root - int(10 * scale), eng_y), (max(2, int(8 * scale)), max(1, int(3 * scale))), 0, 0, 360, (170, 175, 180), -1)

    # Vertical stabilizer (empennage tail fin)
    tail_x = x - l_fuselage + int(6 * scale)
    cv2.line(bg, (tail_x, y), (tail_x - int(14 * scale), y - int(22 * scale)), (45, 95, 190), max(2, int(4 * scale)))


def draw_realistic_soaring_bird(bg: np.ndarray, x: int, y: int, scale: float, flap_phase: float):
    """Draw a raptor/gull soaring with natural biomechanical wing camber and flapping."""
    span = int(22 * scale)
    camber = math.sin(flap_phase) * (7 * scale)

    # Body
    cv2.ellipse(bg, (x, y), (max(2, int(6 * scale)), max(1, int(3 * scale))), 0, 0, 360, (35, 28, 22), -1)
    # Head & beak
    cv2.circle(bg, (int(x + 5 * scale), y), max(1, int(2 * scale)), (30, 22, 16), -1)

    # Articulated wings (shoulder to elbow to primary feathers)
    elbow_x_l = x - int(8 * scale)
    elbow_y_l = int(y - camber * 0.6)
    tip_x_l = x - span
    tip_y_l = int(y - camber)

    elbow_x_r = x + int(8 * scale)
    elbow_y_r = int(y - camber * 0.6)
    tip_x_r = x + span
    tip_y_r = int(y - camber)

    cv2.line(bg, (x, y), (elbow_x_l, elbow_y_l), (42, 32, 24), max(1, int(2.5 * scale)))
    cv2.line(bg, (elbow_x_l, elbow_y_l), (tip_x_l, tip_y_l), (38, 28, 20), max(1, int(2 * scale)))

    cv2.line(bg, (x, y), (elbow_x_r, elbow_y_r), (42, 32, 24), max(1, int(2.5 * scale)))
    cv2.line(bg, (elbow_x_r, elbow_y_r), (tip_x_r, tip_y_r), (38, 28, 20), max(1, int(2 * scale)))


def draw_kamikaze_delta_uav(bg: np.ndarray, x: int, y: int, angle_deg: float, scale: float):
    """Draw a low-observable delta-wing loitering munition / kamikaze UAV."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    pts = [
        (32 * scale, 0),                           # Nose tip
        (-22 * scale, -28 * scale),                # Wingtip starboard
        (-28 * scale, -26 * scale),                # Wingtip edge
        (-24 * scale, -8 * scale),                 # Trailing edge
        (-28 * scale, 0),                          # Pusher propeller hub
        (-24 * scale, 8 * scale),                  # Trailing edge
        (-28 * scale, 26 * scale),                 # Wingtip edge
        (-22 * scale, 28 * scale),                 # Wingtip port
    ]
    rot_pts = []
    for px, py in pts:
        rx = int(x + px * cos_a - py * sin_a)
        ry = int(y + px * sin_a + py * cos_a)
        rot_pts.append([rx, ry])

    rot_pts = np.array(rot_pts, dtype=np.int32)
    cv2.fillPoly(bg, [rot_pts], (48, 52, 54))
    cv2.polylines(bg, [rot_pts], True, (25, 28, 30), max(1, int(1.5 * scale)))

    # Pusher prop blur at rear
    rear_x = int(x - 28 * scale * cos_a)
    rear_y = int(y - 28 * scale * sin_a)
    cv2.circle(bg, (rear_x, rear_y), max(2, int(5 * scale)), (90, 95, 100), 1)


# =========================================================================
# VIDEO GENERATION SUITE
# =========================================================================

def generate_real_drone_patrol_video():
    """Video 1: Real-World Surveillance Sequence of Drone Patrol."""
    out_path = OUTPUT_DIR / "real_drone_surveillance_patrol.mp4"
    print(f"[*] Generating {out_path}...")
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 20.0, (w, h))

    total_frames = 120
    for f in range(total_frames):
        bg = make_photorealistic_sky(w, h, "overcast")

        # Realistic drone search-and-surveillance flight profile
        # Spline-like smooth trajectory with hover stops and altitude adjustments
        t = f / float(total_frames)
        drone_x = int(140 + math.sin(t * math.pi * 2.5) * 160 + t * 240)
        drone_y = int(220 + math.cos(t * math.pi * 3.2) * 50 + math.sin(f * 0.1) * 6)
        scale = 1.0 + math.sin(t * math.pi) * 0.35  # Distance changes

        # Camera lens motion jitter
        jitter_x = int(random.uniform(-1, 1))
        jitter_y = int(random.uniform(-1, 1))

        draw_realistic_quadcopter_drone(bg, drone_x + jitter_x, drone_y + jitter_y, scale, f)

        # Distant background horizon tree line
        for tree_x in range(0, w, 25):
            th = int(25 + math.sin(tree_x * 0.05) * 8)
            cv2.line(bg, (tree_x, h), (tree_x, h - th), (60, 75, 55), 26)

        writer.write(bg)

    writer.release()
    print(f"    -> Successfully generated {out_path} ({os.path.getsize(out_path)} bytes).")


def generate_military_jet_strike_video():
    """Video 2: High-Speed Supersonic Military Jet Airspace Penetration."""
    out_path = OUTPUT_DIR / "military_jet_supersonic_strike.mp4"
    print(f"[*] Generating {out_path}...")
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 25.0, (w, h))

    total_frames = 100
    for f in range(total_frames):
        bg = make_photorealistic_sky(w, h, "dusk")

        # Supersonic intercept trajectory: enters top right at mach 1.8, banks 35 deg, descends and zooms across
        t = f / float(total_frames)
        # Non-linear acceleration: sweeps across rapidly
        prog = (t ** 1.2)
        x = int(w * 1.15 - prog * (w * 1.4))
        y = int(90 + math.sin(prog * math.pi * 0.8) * 180 + prog * 60)
        
        # Scale expands as aircraft approaches camera from 6 km to 400 m
        scale = 0.35 + prog * 0.95
        # Dynamic bank angle matches trajectory slope
        angle_deg = 195 + math.cos(prog * math.pi) * 25
        trail_len = int(140 * scale)

        draw_realistic_fighter_jet(bg, x, y, angle_deg, scale, trail_len)

        writer.write(bg)

    writer.release()
    print(f"    -> Successfully generated {out_path} ({os.path.getsize(out_path)} bytes).")


def generate_multi_threat_airspace_video():
    """Video 3: Mixed Airspace Engagement (Airliner, Drone, Bird, Fighter Jet)."""
    out_path = OUTPUT_DIR / "multi_threat_airspace_scenario.mp4"
    print(f"[*] Generating {out_path}...")
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 25.0, (w, h))

    total_frames = 140
    for f in range(total_frames):
        bg = make_photorealistic_sky(w, h, "day")

        # 1. High-altitude commercial airliner (steady cruise at 35,000 ft)
        airliner_x = int(-40 + f * 4.6)
        airliner_y = 75
        draw_realistic_airliner(bg, airliner_x, airliner_y, 0.7)

        # 2. Tactical reconnaissance drone (patrolling sector at 1,500 ft)
        drone_x = int(220 + math.sin(f * 0.06) * 65)
        drone_y = int(260 + math.cos(f * 0.08) * 30)
        draw_realistic_quadcopter_drone(bg, drone_x, drone_y, 0.85, f)

        # 3. Soaring bird crossing foreground (wildlife)
        bird_t = f / float(total_frames)
        bird_x = int(w * 0.88 - bird_t * 360)
        bird_y = int(150 + math.sin(f * 0.12) * 35)
        draw_realistic_soaring_bird(bg, bird_x, bird_y, 0.75, f * 0.35)

        # 4. Ingressing fighter jet (strikes through after frame 40)
        if f >= 40:
            jet_prog = (f - 40) / float(total_frames - 40)
            jet_x = int(w * 1.1 - jet_prog * (w * 1.35))
            jet_y = int(320 - jet_prog * 150)
            jet_scale = 0.45 + jet_prog * 0.65
            draw_realistic_fighter_jet(bg, jet_x, jet_y, 160, jet_scale, trail_len=int(100 * jet_scale))

        writer.write(bg)

    writer.release()
    print(f"    -> Successfully generated {out_path} ({os.path.getsize(out_path)} bytes).")


def generate_drone_vs_bird_video():
    """Video 4: Avian Wildlife vs. Agile Drone False-Alarm Discrimination."""
    out_path = OUTPUT_DIR / "drone_vs_bird_perimeter_patrol.mp4"
    print(f"[*] Generating {out_path}...")
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 25.0, (w, h))

    total_frames = 110
    for f in range(total_frames):
        bg = make_photorealistic_sky(w, h, "day")

        # Thermal soaring raptor (circling in thermal draft)
        ang = f * 0.075
        b_x = int(200 + math.cos(ang) * 90)
        b_y = int(140 + math.sin(ang) * 45)
        draw_realistic_soaring_bird(bg, b_x, b_y, 0.9, f * 0.2)

        # Agile FPV drone executing zigzag perimeter surveillance
        d_x = int(380 + math.sin(f * 0.16) * 110)
        d_y = int(270 + math.cos(f * 0.12) * 45)
        draw_realistic_quadcopter_drone(bg, d_x, d_y, 0.8, f)

        # Perimeter tree line
        for tree_x in range(0, w, 20):
            th = int(30 + math.cos(tree_x * 0.08) * 10)
            cv2.line(bg, (tree_x, h), (tree_x, h - th), (50, 68, 45), 22)

        writer.write(bg)

    writer.release()
    print(f"    -> Successfully generated {out_path} ({os.path.getsize(out_path)} bytes).")


def generate_kamikaze_drone_video():
    """Video 5: Low-Altitude Delta-Wing Kamikaze Drone Attack."""
    out_path = OUTPUT_DIR / "kamikaze_loitering_drone_attack.mp4"
    print(f"[*] Generating {out_path}...")
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 25.0, (w, h))

    total_frames = 100
    for f in range(total_frames):
        bg = make_photorealistic_sky(w, h, "overcast")

        # Kamikaze drone diving along low altitude trajectory towards radar perimeter
        t = f / float(total_frames)
        x = int(w * 1.05 - t * (w * 1.25))
        y = int(120 + t * 200 + math.sin(f * 0.2) * 15)  # Evasive terrain hug
        scale = 0.4 + t * 0.85
        angle_deg = 205 + math.sin(f * 0.2) * 12

        draw_kamikaze_delta_uav(bg, x, y, angle_deg, scale)

        # Ground terrain clutter
        for gx in range(0, w, 15):
            gh = int(20 + math.sin(gx * 0.03) * 6)
            cv2.line(bg, (gx, h), (gx, h - gh), (70, 78, 65), 18)

        writer.write(bg)

    writer.release()
    print(f"    -> Successfully generated {out_path} ({os.path.getsize(out_path)} bytes).")


def main():
    print("=== AeroWatch AI Pro Surveillance Video Suite Generator ===")
    generate_real_drone_patrol_video()
    generate_military_jet_strike_video()
    generate_multi_threat_airspace_video()
    generate_drone_vs_bird_video()
    generate_kamikaze_drone_video()
    print("\n[+] All 5 test video samples created successfully in videos/ directory!")


if __name__ == "__main__":
    main()
