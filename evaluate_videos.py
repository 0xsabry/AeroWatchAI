"""AeroWatch AI — Tactical Air Defense Video Evaluation & Annotation Engine.

Takes raw surveillance test videos, performs persistent tracking with YOLO,
overlays tactical defense HUDs, radar scopes, and threat telemetry,
and renders annotated production test videos into videos/annotated_*.mp4.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Tactical Color Scheme (BGR)
CLASS_COLORS = {
    0: (0, 140, 255),    # Drone: Tactical Orange
    1: (0, 220, 80),     # Aircraft: Friendly Green
    2: (190, 190, 190),  # Bird: Wildlife Gray
    3: (255, 200, 0),    # Balloon: Sky Cyan
    4: (40, 40, 240),    # Military Jet: Hostile Red
}

THREAT_LABELS = {
    0: ("UNKNOWN UAV", "HIGH"),
    1: ("COMMERCIAL", "FRIENDLY"),
    2: ("WILDLIFE", "NEGLIGIBLE"),
    3: ("AEROSTAT", "MONITOR"),
    4: ("WARPLANE", "CRITICAL"),
}


def draw_tactical_radar_scope(frame: np.ndarray, detections: list[dict], frame_idx: int):
    """Draw a miniature PPI radar scope in the top-right corner with sweeping beam."""
    fh, fw = frame.shape[:2]
    cx = fw - 75
    cy = 75
    radius = 55

    # Radar dark circular backdrop
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius, (15, 25, 15), -1)
    cv2.circle(overlay, (cx, cy), radius, (0, 210, 50), 2)
    cv2.circle(overlay, (cx, cy), int(radius * 0.66), (0, 120, 30), 1)
    cv2.circle(overlay, (cx, cy), int(radius * 0.33), (0, 100, 25), 1)
    # Crosshairs
    cv2.line(overlay, (cx - radius, cy), (cx + radius, cy), (0, 120, 30), 1)
    cv2.line(overlay, (cx, cy - radius), (cx, cy + radius), (0, 120, 30), 1)

    # Sweeping radar beam
    angle = (frame_idx * 6) % 360
    rad = math.radians(angle)
    beam_x = int(cx + radius * math.cos(rad))
    beam_y = int(cy + radius * math.sin(rad))
    cv2.line(overlay, (cx, cy), (beam_x, beam_y), (0, 255, 70), 2)

    # Plot detected targets as radar blips
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Normalize target position to radar radius
        norm_x = ((x1 + x2) / 2.0 / fw) * 2.0 - 1.0
        norm_y = ((y1 + y2) / 2.0 / fh) * 2.0 - 1.0
        bx = int(cx + norm_x * (radius * 0.85))
        by = int(cy + norm_y * (radius * 0.85))

        color = det["color"]
        cv2.circle(overlay, (bx, by), 4, color, -1)
        cv2.circle(overlay, (bx, by), 6, (255, 255, 255), 1)

    # Blend radar scope with frame
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, "RADAR SCOPE", (cx - 45, cy + radius + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 70), 1)


def annotate_frame_tactical(
    frame: np.ndarray,
    results,
    frame_idx: int,
    fps: float,
    track_history: dict[int, list[tuple[int, int]]],
) -> tuple[np.ndarray, list[dict]]:
    """Draw tactical bounding boxes, track vectors, threat badges, and radar scope."""
    fh, fw = frame.shape[:2]
    detections = []
    has_hostile = False
    has_unknown = False

    if results and len(results.boxes) > 0:
        boxes = results.boxes
        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            track_id = int(b.id[0]) if b.id is not None else -1

            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            xc = (x1 + x2) // 2
            yc = (y1 + y2) // 2

            color = CLASS_COLORS.get(cls_id, (0, 255, 255))
            threat_name, threat_level = THREAT_LABELS.get(cls_id, ("UNKNOWN", "MEDIUM"))

            if threat_level == "CRITICAL":
                has_hostile = True
            elif threat_level == "HIGH":
                has_unknown = True

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "cls_id": cls_id,
                "conf": conf,
                "track_id": track_id,
                "color": color,
                "threat_name": threat_name,
                "threat_level": threat_level,
            })

            # 1. Corner brackets HUD bounding box
            corner_len = max(6, min(16, (x2 - x1) // 4))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            # Corner accents
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

            # 2. Tracking centroid & kinematics breadcrumbs
            if track_id != -1:
                track_history.setdefault(track_id, []).append((xc, yc))
                pts = track_history[track_id][-15:]
                for p_idx in range(1, len(pts)):
                    cv2.line(frame, pts[p_idx - 1], pts[p_idx], color, 1)
                cv2.circle(frame, (xc, yc), 3, (255, 255, 255), -1)

            # 3. Tactical Badge Label
            label_text = f"#{track_id} {threat_name} {conf:.0%}" if track_id != -1 else f"{threat_name} {conf:.0%}"
            (tw, th_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + tw + 8, max(20, y1)), color, -1)
            cv2.putText(frame, label_text, (x1 + 4, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    # 4. Top Defense Status Banner
    banner_bg = frame[:34, :].copy()
    cv2.rectangle(banner_bg, (0, 0), (fw, 34), (18, 20, 24), -1)
    cv2.addWeighted(banner_bg, 0.85, frame[:34, :], 0.15, 0, frame[:34, :])

    if has_hostile:
        status_text = "DEFENSE STATUS: RED ALERT — HOSTILE WARPLANE DETECTED"
        status_color = (40, 50, 240)
    elif has_unknown:
        status_text = "DEFENSE STATUS: YELLOW ALERT — UNIDENTIFIED UAV IN SECTOR"
        status_color = (0, 160, 255)
    else:
        status_text = "DEFENSE STATUS: GREEN — AIRSPACE CLEAR"
        status_color = (0, 220, 80)

    cv2.putText(frame, status_text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"AEROWATCH AI  |  FRAME: {frame_idx:04d}  |  FPS: {fps:.1f}", (fw - 380, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 210), 1, cv2.LINE_AA)

    # 5. Radar Scope Overlay
    draw_tactical_radar_scope(frame, detections, frame_idx)

    return frame, detections


def process_video_tactical(video_path: Path, model: YOLO, output_dir: Path) -> dict:
    """Run persistent tracking on a video and render annotated tactical output."""
    out_name = f"annotated_{video_path.name}"
    out_path = output_dir / out_name
    print(f"[*] Processing {video_path.name} -> {out_name}...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    Failed to open {video_path}")
        return {}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    track_history: dict[int, list[tuple[int, int]]] = {}
    detected_frames = 0
    conf_scores = []
    class_detections = {cid: 0 for cid in range(5)}

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Persistent multi-object tracking
        results = model.track(frame, conf=0.30, persist=True, verbose=False)[0]

        annotated_frame, dets = annotate_frame_tactical(frame, results, frame_idx, fps, track_history)
        writer.write(annotated_frame)

        if dets:
            detected_frames += 1
            for d in dets:
                conf_scores.append(d["conf"])
                class_detections[d["cls_id"]] += 1

        frame_idx += 1

    cap.release()
    writer.release()

    # Re-encode to universal H.264 for seamless browser playback
    try:
        import subprocess, shutil, imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        tmp_h264 = out_path.with_name(f"{out_path.stem}_h264_tmp.mp4")
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(out_path),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-movflags", "+faststart",
            "-an",
            str(tmp_h264),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and tmp_h264.exists() and tmp_h264.stat().st_size > 0:
            shutil.move(str(tmp_h264), str(out_path))
    except Exception:
        pass

    elapsed = max(0.001, time.time() - t0)
    process_fps = frame_idx / elapsed

    metrics = {
        "video": video_path.name,
        "total_frames": frame_idx,
        "detected_frames": detected_frames,
        "detection_rate": (detected_frames / frame_idx * 100.0) if frame_idx > 0 else 0.0,
        "mean_conf": (sum(conf_scores) / len(conf_scores) * 100.0) if conf_scores else 0.0,
        "process_fps": process_fps,
        "output_size_bytes": os.path.getsize(out_path) if out_path.exists() else 0,
        "class_detections": class_detections,
    }
    print(f"    -> Rendered {out_name}: {detected_frames}/{frame_idx} frames ({metrics['detection_rate']:.1f}%), Mean Conf: {metrics['mean_conf']:.1f}%, Speed: {process_fps:.1f} FPS")
    return metrics


def main():
    print("=== AeroWatch AI Tactical Video Evaluation Suite ===")
    model_path = Path("weights/aerowatch_v7_balanced.pt")
    if not model_path.exists():
        model_path = Path("weights/aerowatch_v6_expert.pt")
    if not model_path.exists():
        model_path = Path("weights/best.pt")
    print(f"[*] Loading model from {model_path}...")
    model = YOLO(str(model_path))

    video_files = [
        Path("videos/real_drone_surveillance_patrol.mp4"),
        Path("videos/military_jet_supersonic_strike.mp4"),
        Path("videos/multi_threat_airspace_scenario.mp4"),
        Path("videos/drone_vs_bird_perimeter_patrol.mp4"),
        Path("videos/kamikaze_loitering_drone_attack.mp4"),
        Path("videos/drone_real_surveillance.mp4"),
        Path("videos/military_jet_intercept.mp4"),
    ]

    all_metrics = []
    for vf in video_files:
        if vf.exists():
            m = process_video_tactical(vf, model, Path("videos"))
            all_metrics.append(m)

    print("\n=== Video Evaluation Summary ===")
    print(f"{'Video Scenario':<36} | {'Frames':<8} | {'Det %':<8} | {'Mean Conf':<10} | {'FPS':<6} | {'Classes (D/A/B/Bal/J)'}")
    print("-" * 105)
    for m in all_metrics:
        c = m["class_detections"]
        c_str = f"D:{c[0]} A:{c[1]} B:{c[2]} Bal:{c[3]} J:{c[4]}"
        print(f"{m['video']:<36} | {m['total_frames']:<8} | {m['detection_rate']:<7.1f}% | {m['mean_conf']:<9.1f}% | {m['process_fps']:<6.1f} | {c_str}")


if __name__ == "__main__":
    main()
