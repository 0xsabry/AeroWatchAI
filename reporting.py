"""AeroWatch AI — reporting & export helpers (Phase 8).

Pure functions over :class:`history_store.HistoryStore` records:
dashboard metric bundles, CSV / JSON exports, and a Markdown session
report. Everything here describes **unverified model predictions** and
says so in the output itself. No metric in this module claims precision
or recall: those require ground truth and live in the evaluation
documents, not in operational history.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Dict, List, Sequence

from history_store import DetectionRecord, HistoryStore, SessionRecord

PREDICTION_DISCLAIMER = (
    "All detections in this report are unverified computer-vision model "
    "predictions. A detection is not a confirmed identification, threat, "
    "or real-world event. Confidence values are detector scores, not "
    "certainties or probabilities of hostility."
)

SIMULATION_DISCLAIMER = (
    "Interception content in AeroWatch AI is a fictional browser "
    "simulation game. No simulated outcome is a real-world result."
)


def dashboard_metrics(store: HistoryStore) -> Dict:
    """Metric bundle for the dashboard, computed from stored records
    only. Empty history yields zeros and empty collections — the UI is
    responsible for rendering honest empty states."""
    totals = store.totals()
    return {
        "totals": totals,
        "class_distribution": store.class_distribution(),
        "confidence_histogram": store.confidence_histogram(),
        "detections_by_day": store.detections_by_day(),
        "recent_sessions": [s.__dict__ for s in store.list_sessions(limit=5)],
        "has_data": totals["sessions"] > 0,
    }


def detections_to_csv(detections: Sequence[DetectionRecord]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "detection_id", "session_id", "created_at", "canonical_class",
            "display_label", "confidence", "posture", "box_x1", "box_y1",
            "box_x2", "box_y2", "frame_width", "frame_height",
            "verification_status",
        ]
    )
    for d in detections:
        writer.writerow(
            [
                d.detection_id, d.session_id, d.created_at,
                d.canonical_class, d.display_label, f"{d.confidence:.4f}",
                d.posture, *[f"{v:.1f}" for v in d.box_xyxy],
                d.frame_width, d.frame_height, d.verification_status,
            ]
        )
    return buf.getvalue()


def session_to_json(session: SessionRecord, detections: Sequence[DetectionRecord]) -> str:
    payload = {
        "session": session.__dict__,
        "detections": [d.__dict__ for d in detections],
        "disclaimer": PREDICTION_DISCLAIMER,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def history_to_json(store: HistoryStore) -> str:
    sessions = store.list_sessions(limit=1000)
    payload = {
        "sessions": [s.__dict__ for s in sessions],
        "detections": [
            d.__dict__
            for s in sessions
            for d in store.session_detections(s.session_id)
        ],
        "disclaimer": PREDICTION_DISCLAIMER,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def session_report_markdown(
    session: SessionRecord, detections: Sequence[DetectionRecord]
) -> str:
    """A self-contained session report (Markdown, downloadable).

    Includes the identification and disclaimer blocks required for any
    exported AeroWatch report. Simulation disclaimer is included when
    the session notes reference the simulation."""
    lines: List[str] = [
        "# AeroWatch AI — Session Report",
        "",
        f"- **Session ID:** `{session.session_id}`",
        f"- **Kind:** {session.kind}",
        f"- **Started (UTC):** {session.started_at}",
        f"- **Source label:** {session.source_label or '(none)'}",
        f"- **Checkpoint:** {session.checkpoint_name or '(unknown)'}"
        f" (SHA-256 prefix `{session.checkpoint_sha256_prefix or 'n/a'}`)",
        f"- **Runtime backend:** {session.backend or '(unknown)'}",
        f"- **Detections recorded:** {session.detection_count}",
        "",
        "## Class counts",
        "",
    ]
    if session.class_counts:
        for cls, count in sorted(session.class_counts.items()):
            lines.append(f"- {cls}: {count}")
    else:
        lines.append("- (none)")
    lines += ["", "## Detections", ""]
    if detections:
        lines.append(
            "| # | Class | Confidence (detector score) | Posture | Box (x1, y1, x2, y2) | Verification |"
        )
        lines.append("| :- | :- | :- | :- | :- | :- |")
        for i, d in enumerate(detections, 1):
            box = ", ".join(f"{v:.0f}" for v in d.box_xyxy)
            lines.append(
                f"| {i} | {d.canonical_class} | {d.confidence:.3f} | "
                f"{d.posture} | {box} | {d.verification_status} |"
            )
    else:
        lines.append(
            "No individual detection rows stored for this session "
            "(aggregate counts above, e.g. from a video run)."
        )
    lines += ["", "## Disclaimers", "", f"> {PREDICTION_DISCLAIMER}"]
    if "simulat" in (session.notes or "").lower():
        lines.append(f"> {SIMULATION_DISCLAIMER}")
    lines.append("")
    return "\n".join(lines)


def confidence_band(confidence: float, uncertainty_threshold: float) -> str:
    """Explain, in plain language, how a score relates to the
    uncertainty threshold — used by the explainability UI."""
    if confidence < uncertainty_threshold:
        return (
            f"Uncertain: detector score {confidence:.2f} is below the "
            f"uncertainty threshold {uncertainty_threshold:.2f}, so this "
            "prediction is shown as uncertain and needs human review."
        )
    return (
        f"Detector score {confidence:.2f} is at or above the uncertainty "
        f"threshold {uncertainty_threshold:.2f}. It remains an unverified "
        "model prediction — the score is not a certainty."
    )
