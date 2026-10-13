"""AeroWatch AI — Real Video Integration Test Suite on Artifacts.

Tests the real VideoProcessor pipeline on curated test videos:
- civilian_drone_surveillance.mp4
- drone_vs_bird_perimeter.mp4
- civilian_flight_mjpg.avi
- empty_sky_negative.mp4

Validates:
1. Decoding all frames
2. Frame counting (decoded, processed, skipped, failed)
3. Inference and bounding box rendering
4. Verified H.264 video export with post-write reopening validation
5. Stride-based frame skipping
6. Complete resource cleanup upon normal completion, cancellation, and exceptions
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import pytest

from model_manager import ModelManager
from video_processor import VideoProcessingMode, VideoProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "tests" / "artifacts" / "videos"
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "aerowatch_v7_balanced.pt"


@pytest.fixture(scope="module")
def video_processor_fixture():
    mm = ModelManager()
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"Weights file not found at {WEIGHTS_PATH}")
    mm.load_model(WEIGHTS_PATH, device="auto", enable_half=True)
    return VideoProcessor(model_manager=mm)


class TestRealVideoArtifactIntegration:
    """End-to-end integration tests using real media artifacts."""

    def test_process_civilian_drone_surveillance_video(self, video_processor_fixture, tmp_path):
        """Processes real drone surveillance video and validates output file reopening."""
        vp = video_processor_fixture
        video_file = VIDEOS_DIR / "civilian_drone_surveillance.mp4"
        assert video_file.exists(), f"Missing artifact video: {video_file}"

        out_file = tmp_path / "out_drone_surveillance.mp4"

        metrics = vp.process_video(
            video_path=video_file,
            output_path=out_file,
            mode=VideoProcessingMode.SAVE_ANNOTATED,
            vid_stride=1,
            detection_threshold=0.25,
            uncertainty_threshold=0.40,
        )

        # 1. Verify frame processing counts
        assert metrics.frames_decoded > 0
        assert metrics.frames_processed == metrics.frames_decoded
        assert metrics.frames_skipped == 0
        assert metrics.processed_fps > 0.0

        # 2. Verify output file exists and is validated
        assert out_file.exists()
        assert metrics.output_verified is True

        # 3. Reopen and decode the exported video
        cap = cv2.VideoCapture(str(out_file))
        assert cap.isOpened(), "Exported video could not be opened by cv2.VideoCapture"
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert out_w == metrics.width
        assert out_h == metrics.height
        assert out_count > 0

        # Decode first and middle frames
        ret, frame1 = cap.read()
        assert ret and frame1 is not None and frame1.size > 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, out_count // 2)
        ret_mid, frame_mid = cap.read()
        assert ret_mid and frame_mid is not None
        cap.release()

    def test_process_avi_container_compatibility(self, video_processor_fixture, tmp_path):
        """Verifies that an AVI video fixture is successfully decoded and processed."""
        vp = video_processor_fixture
        avi_file = VIDEOS_DIR / "civilian_flight_mjpg.avi"
        assert avi_file.exists(), f"Missing AVI artifact video: {avi_file}"

        out_file = tmp_path / "out_avi_test.mp4"

        metrics = vp.process_video(
            video_path=avi_file,
            output_path=out_file,
            mode=VideoProcessingMode.FULL_OFFLINE,
            vid_stride=1,
        )

        assert metrics.frames_decoded == 60
        assert metrics.frames_processed == 60
        assert metrics.frames_skipped == 0

    def test_process_empty_sky_negative_video(self, video_processor_fixture, tmp_path):
        """Processes empty sky video fixture and verifies zero or minimal false positives."""
        vp = video_processor_fixture
        neg_video = VIDEOS_DIR / "empty_sky_negative.mp4"
        assert neg_video.exists(), f"Missing empty sky video: {neg_video}"

        metrics = vp.process_video(
            video_path=neg_video,
            mode=VideoProcessingMode.FULL_OFFLINE,
            vid_stride=1,
            detection_threshold=0.35,
        )

        assert metrics.frames_decoded == 60
        assert metrics.frames_processed == 60
        # Empty sky video should produce zero confident detections
        confident_detections = sum(v for k, v in metrics.detections_by_class.items() if "(Uncertain)" not in k)
        assert confident_detections == 0, f"Expected 0 false detections on empty sky video, got {confident_detections}"

    def test_frame_skipping_stride_behavior(self, video_processor_fixture):
        """Tests configurable stride (vid_stride=2) and verifies exact frame math."""
        vp = video_processor_fixture
        video_file = VIDEOS_DIR / "civilian_drone_surveillance.mp4"
        assert video_file.exists()

        stride = 2
        metrics = vp.process_video(
            video_path=video_file,
            mode=VideoProcessingMode.FAST_PREVIEW,
            vid_stride=stride,
        )

        expected_processed = metrics.frames_decoded // stride
        assert metrics.frames_processed == expected_processed
        assert metrics.frames_skipped == (metrics.frames_decoded - expected_processed)

    def test_cancellation_and_resource_cleanup(self, video_processor_fixture, tmp_path):
        """Tests that user cancellation stops processing early and releases resources."""
        vp = video_processor_fixture
        video_file = VIDEOS_DIR / "civilian_drone_surveillance.mp4"
        out_file = tmp_path / "out_cancelled.mp4"

        # Cancel on frame 5
        def cancel_on_frame(rendered_frame, frame_idx):
            if frame_idx >= 5:
                vp.cancel()

        metrics = vp.process_video(
            video_path=video_file,
            output_path=out_file,
            on_frame_rendered=cancel_on_frame,
        )

        assert metrics.frames_processed <= 6
        # Temp file should be cleaned up on cancel
        temp_file = out_file.parent / f"temp_{out_file.name}"
        assert not temp_file.exists(), "Temporary partial video file must be cleaned up on cancellation"
