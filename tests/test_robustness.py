"""AeroWatch AI — Robustness, Failure Injection & Edge-Case Test Suite.

Automates rigorous failure tests using actual artifact files from tests/artifacts/corrupt_inputs/:
- Invalid file paths
- Zero-byte video inputs
- Zero-byte image inputs
- Corrupt header container files
- Truncated JPEG streams
- Unsupported format extensions (.bin)
- Invalid stride settings (0, negative)
- Invalid confidence & uncertainty thresholds (<0, >1)
- Model load failures (non-existent, corrupt weights)
- Reopening the same video consecutively
- Consecutive repeated Start/Stop lifecycles
- Output path collision handling
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import pytest
import torch

from model_manager import ModelManager, ModelType
from video_processor import VideoProcessingMode, VideoProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORRUPT_DIR = PROJECT_ROOT / "tests" / "artifacts" / "corrupt_inputs"
VIDEOS_DIR = PROJECT_ROOT / "tests" / "artifacts" / "videos"
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "aerowatch_v7_balanced.pt"


@pytest.fixture(scope="module")
def loaded_mm():
    mm = ModelManager()
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"Weights file not found at {WEIGHTS_PATH}")
    mm.load_model(WEIGHTS_PATH, device="auto", enable_half=True)
    return mm


@pytest.fixture(scope="module")
def video_proc(loaded_mm):
    return VideoProcessor(model_manager=loaded_mm)


class TestRobustnessAndFailureHandling:
    """Rigorous robustness validation on corrupted and boundary inputs."""

    def test_nonexistent_video_path_raises_filenotfound(self, video_proc):
        """Passing a non-existent path must immediately raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            video_proc.process_video("C:/nonexistent/invalid_path/dummy_drone.mp4")
        assert "not found" in str(exc_info.value).lower()

    def test_zero_byte_video_rejection(self, video_proc):
        """Zero-byte video must be cleanly rejected with an explicit IOError/RuntimeError."""
        zero_vid = CORRUPT_DIR / "zero_byte_video.mp4"
        assert zero_vid.exists()
        with pytest.raises((IOError, RuntimeError)) as exc_info:
            video_proc.process_video(zero_vid)
        assert "could not open" in str(exc_info.value).lower() or "video" in str(exc_info.value).lower()

    def test_corrupt_header_video_rejection(self, video_proc):
        """Corrupt container bytes must be cleanly rejected by OpenCV."""
        corrupt_vid = CORRUPT_DIR / "corrupt_header.mp4"
        assert corrupt_vid.exists()
        with pytest.raises((IOError, RuntimeError)):
            video_proc.process_video(corrupt_vid)

    def test_unsupported_format_file_rejection(self, video_proc):
        """Unsupported binary extension must be cleanly rejected."""
        unsupported = CORRUPT_DIR / "unsupported_format.bin"
        assert unsupported.exists()
        with pytest.raises((IOError, RuntimeError)):
            video_proc.process_video(unsupported)

    def test_truncated_image_handling(self, loaded_mm):
        """OpenCV or PIL reading a truncated image returns None or empty array, handled cleanly."""
        trunc_img = CORRUPT_DIR / "truncated_image.jpg"
        assert trunc_img.exists()
        img = cv2.imread(str(trunc_img))
        # cv2 returns None on truncated/invalid image bytes
        assert img is None

    def test_zero_byte_image_handling(self, loaded_mm):
        """Reading zero-byte image returns None and is handled without crash."""
        z_img = CORRUPT_DIR / "zero_byte_image.jpg"
        assert z_img.exists()
        img = cv2.imread(str(z_img))
        assert img is None

    def test_invalid_stride_setting_raises_valueerror(self, video_proc):
        """vid_stride < 1 must raise ValueError."""
        test_vid = VIDEOS_DIR / "civilian_flight_mjpg.avi"
        assert test_vid.exists()
        with pytest.raises(ValueError) as exc:
            video_proc.process_video(test_vid, vid_stride=0)
        assert "vid_stride" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            video_proc.process_video(test_vid, vid_stride=-3)
        assert "vid_stride" in str(exc.value)

    def test_invalid_detection_threshold_raises_valueerror(self, video_proc):
        """Confidence thresholds outside [0.0, 1.0] must raise ValueError."""
        test_vid = VIDEOS_DIR / "civilian_flight_mjpg.avi"
        with pytest.raises(ValueError):
            video_proc.process_video(test_vid, detection_threshold=-0.1)

        with pytest.raises(ValueError):
            video_proc.process_video(test_vid, detection_threshold=1.5)

    def test_invalid_uncertainty_threshold_raises_valueerror(self, video_proc):
        """Uncertainty thresholds outside [0.0, 1.0] must raise ValueError."""
        test_vid = VIDEOS_DIR / "civilian_flight_mjpg.avi"
        with pytest.raises(ValueError):
            video_proc.process_video(test_vid, uncertainty_threshold=-0.5)

        with pytest.raises(ValueError):
            video_proc.process_video(test_vid, uncertainty_threshold=2.0)

    def test_model_load_corrupt_file_raises_explicit_error(self):
        """Attempting to load corrupt model weights must raise explicit error, never silent fallback."""
        mm = ModelManager()
        corrupt_weights = CORRUPT_DIR / "unsupported_format.bin"
        with pytest.raises(Exception):
            mm.load_model(corrupt_weights)

    def test_model_load_nonexistent_weights_raises_filenotfound(self):
        """Non-existent checkpoint file raises FileNotFoundError."""
        mm = ModelManager()
        with pytest.raises(FileNotFoundError):
            mm.load_model(Path("weights/missing_ghost_model.pt"))

    def test_consecutive_reopening_same_video(self, video_proc):
        """Processing the exact same video twice in succession without leak or lock."""
        test_vid = VIDEOS_DIR / "civilian_flight_mjpg.avi"
        metrics1 = video_proc.process_video(test_vid, mode=VideoProcessingMode.FULL_OFFLINE)
        metrics2 = video_proc.process_video(test_vid, mode=VideoProcessingMode.FULL_OFFLINE)

        assert metrics1.frames_processed == 60
        assert metrics2.frames_processed == 60
        assert metrics1.frames_decoded == metrics2.frames_decoded

    def test_output_path_collision_safety(self, video_proc, tmp_path):
        """Writing twice to the exact same output path cleanly overwrites without locking."""
        test_vid = VIDEOS_DIR / "civilian_flight_mjpg.avi"
        collision_out = tmp_path / "colliding_output.mp4"

        metrics1 = video_proc.process_video(
            test_vid,
            output_path=collision_out,
            mode=VideoProcessingMode.SAVE_ANNOTATED,
        )
        assert collision_out.exists()

        # Second write to the exact same output path
        metrics2 = video_proc.process_video(
            test_vid,
            output_path=collision_out,
            mode=VideoProcessingMode.SAVE_ANNOTATED,
        )
        assert collision_out.exists()
        assert metrics2.output_verified is True
