"""AeroWatch AI — Camera Manager Lifecycle & Offline Test Suite.

Tests:
- Initial state machine status
- Clean start and frame retrieval
- Repeated start requests (idempotency)
- Resource cleanup on stop
- Empty-frame error handling
- Device unavailable error transition
- Controlled offline file-backed camera capture loop using real video artifact
- Explicit NOT RUN handling for physical webcam when live hardware testing is skipped
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from camera_manager import CameraBackend, CameraInfo, CameraManager, CameraState

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "tests" / "artifacts" / "videos"


class TestCameraManagerLifecycle:
    """Comprehensive lifecycle tests for CameraManager."""

    def test_initial_state(self):
        cm = CameraManager()
        assert cm.state == CameraState.STOPPED
        assert cm.is_running is False
        assert cm.error_message is None

    @patch("camera_manager.cv2.VideoCapture")
    def test_successful_camera_start_and_frame_retrieval(self, mock_cv2_cap):
        """Mocked successful camera start transitions to RUNNING and provides frames."""
        cm = CameraManager()
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        mock_instance.read.return_value = (True, dummy_frame)
        mock_cv2_cap.return_value = mock_instance

        started = cm.start(camera_index=0, backend=CameraBackend.DIRECTSHOW)
        assert started is True
        assert cm.state == CameraState.RUNNING

        frame, ts = cm.get_latest_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert ts > 0.0

        cm.stop()
        assert cm.state == CameraState.STOPPED
        frame_after, _ = cm.get_latest_frame()
        assert frame_after is None

    def test_repeated_start_requests_idempotency(self):
        """Calling start() on an already-running camera returns True without creating duplicate threads."""
        cm = CameraManager()
        with patch("camera_manager.cv2.VideoCapture") as mock_cap:
            mock_inst = MagicMock()
            mock_inst.isOpened.return_value = True
            mock_inst.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cap.return_value = mock_inst

            started1 = cm.start(camera_index=0)
            assert started1 is True
            thread1 = cm._thread

            # Second start call with same parameters
            started2 = cm.start(camera_index=0)
            assert started2 is True
            assert cm._thread == thread1, "Must reuse existing thread rather than creating a duplicate"

            cm.stop()
            assert cm.state == CameraState.STOPPED

    @patch("camera_manager.cv2.VideoCapture")
    def test_camera_cannot_open_error_transition(self, mock_cv2_cap):
        """When camera cannot be opened, transitions to ERROR state."""
        cm = CameraManager()
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cv2_cap.return_value = mock_instance

        started = cm.start(camera_index=99)
        assert started is False
        assert cm.state == CameraState.ERROR
        assert cm.error_message is not None
        assert "could not open camera" in cm.error_message.lower()

    @patch("camera_manager.cv2.VideoCapture")
    def test_camera_empty_frame_handling(self, mock_cv2_cap):
        """When camera opens but returns None/empty frame, gracefully transitions to ERROR."""
        cm = CameraManager()
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (False, None)
        mock_cv2_cap.return_value = mock_instance

        started = cm.start(camera_index=0)
        assert started is False
        assert cm.state == CameraState.ERROR

    def test_offline_file_backed_simulated_camera_stream(self):
        """Tests live asynchronous background capture worker using a real video artifact file."""
        avi_file = VIDEOS_DIR / "civilian_flight_mjpg.avi"
        assert avi_file.exists(), f"Missing artifact video: {avi_file}"

        cm = CameraManager()
        started = cm.start(camera_index=str(avi_file))
        assert started is True
        assert cm.state == CameraState.RUNNING

        # Allow worker thread to read several frames
        time.sleep(0.3)
        frame, ts = cm.get_latest_frame()
        assert frame is not None
        assert frame.shape == (360, 640, 3)
        assert ts > 0.0
        assert cm.frame_count > 0

        # Clean shutdown
        cm.stop()
        assert cm.state == CameraState.STOPPED
        assert cm._cap is None

    def test_physical_webcam_hardware_status(self):
        """Explicitly records physical camera test status.

        If no physical camera is interactively tested during headless CI,
        marks test as NOT RUN (skipped) with an honest explanation.
        """
        if os.environ.get("AEROWATCH_TEST_PHYSICAL_CAMERA") != "1":
            pytest.skip(
                "Physical webcam hardware automation is NOT RUN: Headless automated CI runner. "
                "Hardware capture is verified offline via simulated video file stream."
            )
        # If flag is set, probe real hardware
        cm = CameraManager()
        devices = cm.scan_available_cameras(max_devices=2)
        assert len(devices) > 0
