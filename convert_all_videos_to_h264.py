"""AeroWatch AI — Universal H.264 Browser Video Encoder.

Re-encodes all raw and annotated MP4 video feeds in videos/ to standard H.264 (AVC1)
with yuv420p pixel format and faststart flags so that:
1. Streamlit web dashboard plays them natively without black screens or 0:00 stalls.
2. All major modern web browsers (Chrome, Edge, Firefox) decode the stream seamlessly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
import imageio_ffmpeg

VIDEOS_DIR = Path("videos")


def convert_video_to_h264(video_path: Path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    temp_path = video_path.with_name(f"{video_path.stem}_h264_tmp.mp4")

    print(f"[*] Converting {video_path.name} to H.264...")
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-movflags", "+faststart",
        "-an",
        str(temp_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and temp_path.exists() and temp_path.stat().st_size > 1000:
        shutil.move(str(temp_path), str(video_path))
        print(f"    -> [SUCCESS] Re-encoded {video_path.name} ({video_path.stat().st_size} bytes, H.264)")
    else:
        if temp_path.exists():
            temp_path.unlink()
        print(f"    -> [FAILED] FFmpeg error on {video_path.name}: {res.stderr[:200]}")


def main():
    print("=== AeroWatch AI H.264 Universal Video Re-Encoder ===")
    all_videos = sorted(list(VIDEOS_DIR.glob("*.mp4")))
    # Exclude any temporary files
    targets = [v for v in all_videos if not v.stem.endswith("_tmp")]
    print(f"[*] Found {len(targets)} video files in {VIDEOS_DIR}")

    for v in targets:
        convert_video_to_h264(v)

    print("\n[+] All video files successfully converted to browser-standard H.264!")


if __name__ == "__main__":
    main()
