"""AeroWatch AI — Resilient Camera Manager & Hardware Capture Lifecycle.

Provides:
- Strict state machine: STOPPED -> OPENING -> RUNNING -> ERROR -> STOPPED
- Windows backend support: DirectShow (CAP_DSHOW), Media Foundation (CAP_MSMF), Default (CAP_ANY)
- Camera diagnostic and discovery routine (checks indices and validates frame reads)
- Dedicated background capture thread with short buffer (CAP_PROP_BUFFERSIZE=1) to prevent display lag
- Thread-safe latest-frame atomic buffer
- Graceful error handling for device busy, disconnected, or empty frames
- Clean resource release on exit without device lockout
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("AeroWatch.CameraManager")


class CameraState(str, Enum):
    STOPPED = "STOPPED"
    OPENING = "OPENING"
    RUNNING = "RUNNING"
    ERROR = "ERROR"


class CameraBackend(str, Enum):
    DIRECTSHOW = "DirectShow (Windows Recommended)"
    MEDIA_FOUNDATION = "Media Foundation (Windows)"
    DEFAULT = "Default (Auto)"

    @property
    def cv2_constant(self) -> int:
        if self == CameraBackend.DIRECTSHOW:
            return cv2.CAP_DSHOW
        elif self == CameraBackend.MEDIA_FOUNDATION:
            return cv2.CAP_MSMF
        return cv2.CAP_ANY


@dataclass
class CameraInfo:
    index: int
    backend: CameraBackend
    is_available: bool
    can_read: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    error_message: Optional[str] = None


class CameraManager:
    """Manages thread-safe camera capture, backend fallback, and latest-frame retrieval."""

    def __init__(self):
        self._state: CameraState = CameraState.STOPPED
        self._error_msg: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Atomic frame store
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_count: int = 0
        self._fps_measured: float = 0.0

        # Configuration
        self._camera_index: int = 0
        self._backend: CameraBackend = CameraBackend.DIRECTSHOW
        self._requested_width: int = 640
        self._requested_height: int = 480
        self._target_fps: int = 30

    @property
    def state(self) -> CameraState:
        with self._lock:
            return self._state

    @property
    def error_message(self) -> Optional[str]:
        with self._lock:
            return self._error_msg

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._state == CameraState.RUNNING

    @property
    def measured_fps(self) -> float:
        with self._lock:
            return self._fps_measured

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @staticmethod
    def probe_available_cameras(max_indices: int = 3) -> List[CameraInfo]:
        """Probes camera indices using Windows DirectShow and Media Foundation backends."""
        results: List[CameraInfo] = []
        backends_to_test = [CameraBackend.DIRECTSHOW, CameraBackend.MEDIA_FOUNDATION, CameraBackend.DEFAULT]

        for idx in range(max_indices):
            found_any = False
            for backend in backends_to_test:
                cap = None
                try:
                    cap = cv2.VideoCapture(idx, backend.cv2_constant)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                            results.append(
                                CameraInfo(
                                    index=idx,
                                    backend=backend,
                                    is_available=True,
                                    can_read=True,
                                    width=w,
                                    height=h,
                                    fps=fps,
                                )
                            )
                            found_any = True
                            cap.release()
                            break  # Found best backend for this index
                        else:
                            results.append(
                                CameraInfo(
                                    index=idx,
                                    backend=backend,
                                    is_available=True,
                                    can_read=False,
                                    error_message="Camera opened but returned empty or corrupt frame.",
                                )
                            )
                    cap.release()
                except Exception as e:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
            if not found_any and not any(r.index == idx for r in results):
                results.append(
                    CameraInfo(
                        index=idx,
                        backend=CameraBackend.DIRECTSHOW,
                        is_available=False,
                        can_read=False,
                        error_message="Device index unavailable or in use by another application.",
                    )
                )

        return results

    def start(
        self,
        camera_index: Union[int, str] = 0,
        backend: CameraBackend = CameraBackend.DIRECTSHOW,
        width: int = 640,
        height: int = 480,
        target_fps: int = 30,
    ) -> bool:
        """Starts asynchronous camera capture on the background thread."""
        with self._lock:
            if self._state == CameraState.RUNNING:
                if self._camera_index == camera_index and self._backend == backend:
                    logger.info("Camera already running with requested parameters.")
                    return True
                self._stop_locked()

            self._state = CameraState.OPENING
            self._error_msg = None
            self._camera_index = camera_index
            self._backend = backend
            self._requested_width = width
            self._requested_height = height
            self._target_fps = target_fps
            self._stop_event.clear()
            self._latest_frame = None
            self._frame_count = 0

        self._thread = threading.Thread(target=self._capture_worker, daemon=True, name="AeroWatch-CameraCapture")
        self._thread.start()

        # Wait briefly for worker to initialize
        t0 = time.time()
        while time.time() - t0 < 3.0:
            with self._lock:
                if self._state == CameraState.RUNNING:
                    return True
                if self._state == CameraState.ERROR:
                    return False
            time.sleep(0.05)

        with self._lock:
            return self._state == CameraState.RUNNING

    def _capture_worker(self):
        """Dedicated background thread keeping buffer size=1 for lowest display latency."""
        backend_order = [self._backend]
        # Add fallback backends if requested backend fails
        for b in [CameraBackend.DIRECTSHOW, CameraBackend.MEDIA_FOUNDATION, CameraBackend.DEFAULT]:
            if b not in backend_order:
                backend_order.append(b)

        cap = None
        for b in backend_order:
            try:
                if isinstance(self._camera_index, str):
                    logger.info(f"Opening file-backed simulated camera source: {self._camera_index}")
                    cap = cv2.VideoCapture(self._camera_index)
                else:
                    logger.info(f"Attempting to open camera {self._camera_index} with {b.value}...")
                    cap = cv2.VideoCapture(self._camera_index, b.cv2_constant)
                if cap.isOpened():
                    # Set short buffer to minimize display lag
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
                    cap.set(cv2.CAP_PROP_FPS, self._target_fps)

                    # Validate first frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        logger.info(f"Successfully started camera on {b.value}")
                        self._cap = cap
                        with self._lock:
                            self._state = CameraState.RUNNING
                            self._backend = b
                            self._latest_frame = test_frame.copy()
                            self._latest_timestamp = time.time()
                            self._frame_count = 1
                        break
                    else:
                        cap.release()
                        cap = None
            except Exception as e:
                logger.warning(f"Failed backend {b.value}: {e}")
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cap = None

        if cap is None or not cap.isOpened():
            with self._lock:
                self._state = CameraState.ERROR
                self._error_msg = (
                    f"Could not open camera {self._camera_index}. Device may be in use by another app "
                    f"or permission is denied. Tested backends: {[b.name for b in backend_order]}."
                )
            logger.error(self._error_msg)
            return

        # Main capture loop
        fps_timer = time.time()
        fps_counter = 0

        consecutive_failures = 0
        while not self._stop_event.is_set():
            try:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures > 30:  # ~1 second of continuous drop
                        with self._lock:
                            self._state = CameraState.ERROR
                            self._error_msg = "Camera disconnected or stopped returning video frames."
                        break
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0
                now = time.time()
                with self._lock:
                    self._latest_frame = frame
                    self._latest_timestamp = now
                    self._frame_count += 1

                fps_counter += 1
                if now - fps_timer >= 1.0:
                    with self._lock:
                        self._fps_measured = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now

            except Exception as e:
                logger.error(f"Exception in camera capture loop: {e}")
                with self._lock:
                    self._state = CameraState.ERROR
                    self._error_msg = f"Camera capture error: {e}"
                break

        # Teardown
        try:
            cap.release()
        except Exception:
            pass
        self._cap = None
        with self._lock:
            if self._state != CameraState.ERROR:
                self._state = CameraState.STOPPED

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Returns (latest_bgr_frame, timestamp). Thread-safe and non-blocking."""
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_timestamp

    def stop(self):
        """Stops the camera capture thread and releases video resources cleanly."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self):
        self._stop_event.set()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._state = CameraState.STOPPED
        self._latest_frame = None
