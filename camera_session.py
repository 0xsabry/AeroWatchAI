"""AeroWatch AI — camera session lifecycle tracker (Phase 9).

Pure aggregation logic for one camera streaming session, deliberately
separate from Streamlit and from the camera hardware so the lifecycle
can be unit-tested with simulated detections. **No frames or images
are ever stored** — only counts, durations, and posture tallies.

Lifecycle: ``start()`` → ``observe(detections)`` per processed frame →
``stop()`` (clean end) or ``interrupt()`` (failure / killed run) →
``summary()``. A session that is never explicitly stopped is reported
by ``summary()`` with ``ended_cleanly=False`` once finalised by the
caller, so interrupted sessions are recorded honestly instead of being
silently lost or silently treated as clean.

Aggregates are *actual counts of model predictions observed during the
session*. They are not tracks: the same physical object counted in 30
frames contributes 30 to the tally, and the summary says so.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, Iterable, Optional


class CameraSessionTracker:
    def __init__(self, source_label: str = "camera") -> None:
        self.session_id = uuid.uuid4().hex[:16]
        self.source_label = source_label
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.frames_processed = 0
        self.class_counts: Dict[str, int] = {}
        self.posture_counts: Dict[str, int] = {}
        self.ended_cleanly = False
        self._active = False

    # -- lifecycle ---------------------------------------------------

    def start(self, *, now: Optional[float] = None) -> None:
        self.started_at = time.time() if now is None else now
        self._active = True

    @property
    def is_active(self) -> bool:
        return self._active

    def observe(self, detections: Iterable[Dict]) -> None:
        """Fold one processed frame's validated detections into the
        aggregates. Malformed entries are skipped, never invented."""
        if not self._active:
            return
        self.frames_processed += 1
        for det in detections:
            try:
                cls = str(det.get("internal_name") or det.get("canonical_class") or "unknown")
            except AttributeError:
                continue
            self.class_counts[cls] = self.class_counts.get(cls, 0) + 1
            posture = det.get("posture")
            if posture:
                key = str(getattr(posture, "value", posture))
                self.posture_counts[key] = self.posture_counts.get(key, 0) + 1

    def stop(self, *, now: Optional[float] = None) -> Dict:
        self.ended_at = time.time() if now is None else now
        self.ended_cleanly = True
        self._active = False
        return self.summary()

    def interrupt(self, *, now: Optional[float] = None) -> Dict:
        self.ended_at = time.time() if now is None else now
        self.ended_cleanly = False
        self._active = False
        return self.summary()

    # -- reporting -----------------------------------------------------

    def summary(self, *, now: Optional[float] = None) -> Dict:
        end = self.ended_at
        if end is None:
            end = time.time() if now is None else now
        duration = max(0.0, end - self.started_at) if self.started_at else 0.0
        return {
            "tracker_session_id": self.session_id,
            "source_label": self.source_label,
            "duration_sec": round(duration, 1),
            "frames_processed": self.frames_processed,
            "class_counts": dict(self.class_counts),
            "posture_counts": dict(self.posture_counts),
            "total_detections": sum(self.class_counts.values()),
            "ended_cleanly": self.ended_cleanly,
            "counting_note": (
                "Counts are per-frame model predictions aggregated over "
                "the session; one physical object seen in many frames is "
                "counted many times. This is not object tracking."
            ),
        }
