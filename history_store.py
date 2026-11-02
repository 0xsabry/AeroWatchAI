"""AeroWatch AI — persistent detection-history store (Phase 8).

A local, single-user record of *processing sessions* and the model
predictions they produced. Design rules (product + privacy):

- **Metadata only.** The store keeps class labels, detector scores,
  postures, box geometry, timestamps, and model/run identification. It
  never stores image or video bytes, never stores full file paths
  (only a basename label), and never stores operator identities.
- **Predictions, not facts.** Every detection row is an unverified
  model prediction; ``verification_status`` defaults to ``unverified``
  and is only ever changed by an explicit human action in the UI.
- **History, not tracking.** A session groups predictions from one
  processing run. Nothing here associates objects across sessions or
  implies persistent real-world object tracking.
- **Local only.** SQLite file on the user's machine (default
  ``runs/history/aerowatch_history.db``, override with the
  ``AEROWATCH_HISTORY_DIR`` environment variable). No network access.
- **Isolation.** All state lives behind a ``HistoryStore`` instance
  bound to one database file; two instances on different files share
  nothing. The application uses one instance for its local database.
- **Retention & deletion.** ``prune(retention_days)`` deletes sessions
  older than the configured retention; ``delete_session`` and
  ``clear_all`` provide explicit, immediate deletion.

This module is deliberately independent of Streamlit so it can be unit
tested without a UI.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SCHEMA_VERSION = 1
DEFAULT_RETENTION_DAYS = 30
MAX_DETECTIONS_PER_SESSION = 5000

SESSION_KINDS = ("image", "video", "camera", "benchmark", "other")


def default_history_path() -> Path:
    """Resolve the history database path (env override, else runs/)."""
    override = os.environ.get("AEROWATCH_HISTORY_DIR")
    base = Path(override) if override else Path(__file__).resolve().parent / "runs" / "history"
    return base / "aerowatch_history.db"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SessionRecord:
    session_id: str
    kind: str
    started_at: str
    source_label: str
    checkpoint_name: str
    checkpoint_sha256_prefix: str
    backend: str
    detection_count: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    notes: str = ""


@dataclass
class DetectionRecord:
    detection_id: int
    session_id: str
    created_at: str
    canonical_class: str
    display_label: str
    confidence: float
    posture: str
    box_xyxy: List[float]
    frame_width: int
    frame_height: int
    verification_status: str = "unverified"


class HistoryStore:
    """SQLite-backed store for processing sessions and predictions."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else default_history_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # One connection per thread: Streamlit serves sessions from
        # multiple threads and shares cached resources across them, so
        # a single sqlite3 connection object is not safe to share.
        # WAL mode makes per-thread connections to one file sound.
        self._local = threading.local()
        self._create_schema()

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    # -- lifecycle ---------------------------------------------------

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self) -> "HistoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                started_at TEXT NOT NULL,
                source_label TEXT NOT NULL DEFAULT '',
                checkpoint_name TEXT NOT NULL DEFAULT '',
                checkpoint_sha256_prefix TEXT NOT NULL DEFAULT '',
                backend TEXT NOT NULL DEFAULT '',
                detection_count INTEGER NOT NULL DEFAULT 0,
                class_counts_json TEXT NOT NULL DEFAULT '{}',
                notes TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id)
                    ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                canonical_class TEXT NOT NULL,
                display_label TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL,
                posture TEXT NOT NULL DEFAULT '',
                box_x1 REAL NOT NULL DEFAULT 0,
                box_y1 REAL NOT NULL DEFAULT 0,
                box_x2 REAL NOT NULL DEFAULT 0,
                box_y2 REAL NOT NULL DEFAULT 0,
                frame_width INTEGER NOT NULL DEFAULT 0,
                frame_height INTEGER NOT NULL DEFAULT 0,
                verification_status TEXT NOT NULL DEFAULT 'unverified'
            );
            CREATE INDEX IF NOT EXISTS idx_det_session
                ON detections(session_id);
            CREATE INDEX IF NOT EXISTS idx_det_class
                ON detections(canonical_class);
            CREATE INDEX IF NOT EXISTS idx_sessions_started
                ON sessions(started_at);
            """
        )
        self._conn.commit()

    # -- writing -----------------------------------------------------

    def start_session(
        self,
        *,
        kind: str,
        source_label: str = "",
        checkpoint_name: str = "",
        checkpoint_sha256: str = "",
        backend: str = "",
        notes: str = "",
        started_at: Optional[datetime] = None,
    ) -> str:
        """Open a processing-session record; returns its session id."""
        if kind not in SESSION_KINDS:
            kind = "other"
        session_id = uuid.uuid4().hex[:16]
        # Only a basename is ever stored — never a path.
        label = Path(str(source_label)).name if source_label else ""
        self._conn.execute(
            "INSERT INTO sessions (session_id, kind, started_at,"
            " source_label, checkpoint_name, checkpoint_sha256_prefix,"
            " backend, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                kind,
                _iso(started_at or _utc_now()),
                label[:200],
                str(checkpoint_name)[:200],
                str(checkpoint_sha256)[:12],
                str(backend)[:50],
                str(notes)[:500],
            ),
        )
        self._conn.commit()
        return session_id

    def add_detections(
        self,
        session_id: str,
        detections: Sequence[Dict],
        *,
        frame_width: int = 0,
        frame_height: int = 0,
        created_at: Optional[datetime] = None,
    ) -> int:
        """Append prediction rows to a session. Returns rows stored.

        Each detection dict accepts the keys produced by
        ``ModelManager.filter_airspace_detections`` plus a posture:
        ``internal_name``/``canonical_class``, ``display_label``,
        ``conf``/``confidence``, ``posture``, ``box``/``box_xyxy``.
        Rows are capped per session (``MAX_DETECTIONS_PER_SESSION``)
        so a long video cannot grow the database without bound.
        Malformed rows are skipped, never fabricated.
        """
        existing = self._session_or_raise(session_id)
        budget = MAX_DETECTIONS_PER_SESSION - int(existing["detection_count"])
        if budget <= 0:
            return 0
        stamp = _iso(created_at or _utc_now())
        stored = 0
        counts: Dict[str, int] = json.loads(existing["class_counts_json"] or "{}")
        for det in list(detections)[:budget]:
            try:
                canonical = str(det.get("internal_name") or det.get("canonical_class") or "unknown")
                conf = float(det.get("conf", det.get("confidence")))
                if not (0.0 <= conf <= 1.0):
                    continue
                box = det.get("box") or det.get("box_xyxy") or [0, 0, 0, 0]
                x1, y1, x2, y2 = (float(v) for v in list(box)[:4])
            except (TypeError, ValueError):
                continue
            self._conn.execute(
                "INSERT INTO detections (session_id, created_at,"
                " canonical_class, display_label, confidence, posture,"
                " box_x1, box_y1, box_x2, box_y2, frame_width,"
                " frame_height) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    stamp,
                    canonical[:100],
                    str(det.get("display_label", canonical))[:100],
                    conf,
                    str(det.get("posture", ""))[:50],
                    x1, y1, x2, y2,
                    int(frame_width),
                    int(frame_height),
                ),
            )
            counts[canonical] = counts.get(canonical, 0) + 1
            stored += 1
        if stored:
            self._conn.execute(
                "UPDATE sessions SET detection_count = detection_count + ?,"
                " class_counts_json = ? WHERE session_id = ?",
                (stored, json.dumps(counts, sort_keys=True), session_id),
            )
            self._conn.commit()
        return stored

    def record_class_counts(
        self,
        session_id: str,
        class_counts: Dict[str, int],
    ) -> None:
        """Record aggregate per-class counts for a session whose
        individual predictions are not stored (e.g. a long video run,
        where the processor reports totals). Counts are labelled as
        aggregates on the session; no detection rows are created."""
        existing = self._session_or_raise(session_id)
        counts = {str(k)[:100]: int(v) for k, v in class_counts.items() if int(v) > 0}
        total = sum(counts.values())
        self._conn.execute(
            "UPDATE sessions SET detection_count = ?, class_counts_json = ?"
            " WHERE session_id = ?",
            (total, json.dumps(counts, sort_keys=True), session_id),
        )
        self._conn.commit()
        _ = existing

    # -- reading -----------------------------------------------------

    def _session_or_raise(self, session_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown session: {session_id}")
        return row

    @staticmethod
    def _to_session(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            kind=row["kind"],
            started_at=row["started_at"],
            source_label=row["source_label"],
            checkpoint_name=row["checkpoint_name"],
            checkpoint_sha256_prefix=row["checkpoint_sha256_prefix"],
            backend=row["backend"],
            detection_count=int(row["detection_count"]),
            class_counts=json.loads(row["class_counts_json"] or "{}"),
            notes=row["notes"],
        )

    @staticmethod
    def _to_detection(row: sqlite3.Row) -> DetectionRecord:
        return DetectionRecord(
            detection_id=int(row["detection_id"]),
            session_id=row["session_id"],
            created_at=row["created_at"],
            canonical_class=row["canonical_class"],
            display_label=row["display_label"],
            confidence=float(row["confidence"]),
            posture=row["posture"],
            box_xyxy=[row["box_x1"], row["box_y1"], row["box_x2"], row["box_y2"]],
            frame_width=int(row["frame_width"]),
            frame_height=int(row["frame_height"]),
            verification_status=row["verification_status"],
        )

    def list_sessions(self, *, limit: int = 100) -> List[SessionRecord]:
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [self._to_session(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return self._to_session(row) if row else None

    def query_detections(
        self,
        *,
        classes: Optional[Sequence[str]] = None,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        search: str = "",
        session_id: Optional[str] = None,
        limit: int = 500,
    ) -> List[DetectionRecord]:
        """Filtered detection search across sessions.

        ``search`` matches (case-insensitive substring) against the
        canonical class, display label, and the owning session's source
        label. Dates are ISO strings compared lexicographically, which
        is correct for the UTC ISO format this store writes.
        """
        clauses = ["d.confidence >= ?", "d.confidence <= ?"]
        params: List = [float(min_confidence), float(max_confidence)]
        if classes:
            placeholders = ",".join("?" for _ in classes)
            clauses.append(f"lower(d.canonical_class) IN ({placeholders})")
            params.extend(str(c).lower() for c in classes)
        if date_from:
            clauses.append("d.created_at >= ?")
            params.append(str(date_from))
        if date_to:
            clauses.append("d.created_at <= ?")
            params.append(str(date_to))
        if session_id:
            clauses.append("d.session_id = ?")
            params.append(session_id)
        if search:
            clauses.append(
                "(lower(d.canonical_class) LIKE ? OR lower(d.display_label)"
                " LIKE ? OR lower(s.source_label) LIKE ?)"
            )
            needle = f"%{search.lower()}%"
            params.extend([needle, needle, needle])
        sql = (
            "SELECT d.* FROM detections d JOIN sessions s"
            " ON d.session_id = s.session_id WHERE "
            + " AND ".join(clauses)
            + " ORDER BY d.created_at DESC, d.detection_id DESC LIMIT ?"
        )
        params.append(int(limit))
        rows = self._conn.execute(sql, params).fetchall()
        return [self._to_detection(r) for r in rows]

    def session_detections(self, session_id: str) -> List[DetectionRecord]:
        rows = self._conn.execute(
            "SELECT * FROM detections WHERE session_id = ?"
            " ORDER BY detection_id ASC",
            (session_id,),
        ).fetchall()
        return [self._to_detection(r) for r in rows]

    # -- verification (explicit human action only) -------------------

    def set_verification(self, detection_id: int, status: str) -> None:
        if status not in ("unverified", "verified", "rejected"):
            raise ValueError(f"invalid verification status: {status!r}")
        self._conn.execute(
            "UPDATE detections SET verification_status = ?"
            " WHERE detection_id = ?",
            (status, int(detection_id)),
        )
        self._conn.commit()

    # -- deletion & retention ----------------------------------------

    def delete_session(self, session_id: str) -> bool:
        self._conn.execute(
            "DELETE FROM detections WHERE session_id = ?", (session_id,)
        )
        cur = self._conn.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def clear_all(self) -> None:
        self._conn.execute("DELETE FROM detections")
        self._conn.execute("DELETE FROM sessions")
        self._conn.commit()

    def prune(self, retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
        """Delete sessions started more than ``retention_days`` ago.
        Returns the number of sessions removed."""
        cutoff = _iso(_utc_now() - timedelta(days=int(retention_days)))
        ids = [
            r["session_id"]
            for r in self._conn.execute(
                "SELECT session_id FROM sessions WHERE started_at < ?",
                (cutoff,),
            ).fetchall()
        ]
        for sid in ids:
            self.delete_session(sid)
        return len(ids)

    # -- aggregates (dashboard / analytics) --------------------------

    def totals(self) -> Dict:
        sessions = self._conn.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"]
        detections = self._conn.execute("SELECT COUNT(*) c FROM detections").fetchone()["c"]
        unverified = self._conn.execute(
            "SELECT COUNT(*) c FROM detections WHERE verification_status = 'unverified'"
        ).fetchone()["c"]
        aggregate_only = self._conn.execute(
            "SELECT COALESCE(SUM(detection_count), 0) c FROM sessions"
        ).fetchone()["c"]
        return {
            "sessions": int(sessions),
            "stored_detections": int(detections),
            "unverified_detections": int(unverified),
            "total_detections_including_aggregates": int(aggregate_only),
        }

    def class_distribution(self) -> Dict[str, int]:
        rows = self._conn.execute(
            "SELECT canonical_class, COUNT(*) c FROM detections"
            " GROUP BY canonical_class ORDER BY c DESC"
        ).fetchall()
        return {r["canonical_class"]: int(r["c"]) for r in rows}

    def confidence_histogram(self, *, bins: int = 10) -> List[int]:
        counts = [0] * bins
        for row in self._conn.execute("SELECT confidence FROM detections"):
            idx = min(bins - 1, int(float(row["confidence"]) * bins))
            counts[idx] += 1
        return counts

    def detections_by_day(self) -> Dict[str, int]:
        rows = self._conn.execute(
            "SELECT substr(created_at, 1, 10) day, COUNT(*) c FROM detections"
            " GROUP BY day ORDER BY day ASC"
        ).fetchall()
        return {r["day"]: int(r["c"]) for r in rows}

    @staticmethod
    def content_fingerprint(payload: str) -> str:
        """Stable short fingerprint used by the UI to avoid recording
        the same analysis twice on Streamlit reruns."""
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
