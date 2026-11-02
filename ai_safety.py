"""AeroWatch AI — AI Safety & Detection Contract Layer.

This module is the single validation boundary between raw model output /
user input and every downstream consumer (rendering, metrics, threat
presentation, and the engagement *simulation*).

Design rules enforced here:

1. **Model output is untrusted input.** A detection is only usable after it
   passes the contract in :func:`sanitize_detections`: finite confidence in
   ``[0, 1]``, finite box coordinates, positive area, and coordinates clamped
   to the frame. Malformed detections are dropped with a counted reason —
   never silently repaired into a plausible-looking target, and never allowed
   to crash the pipeline (previously ``int(NaN)`` raised ``ValueError`` and
   killed frame processing, while a ``NaN`` confidence *passed* the old
   ``confidence < threshold`` check and was displayed as a real target).

2. **Classification is not hostility.** The detector's civilian classes
   (``airplane`` = Aircraft, ``bird``, ``balloon``) are *non-hostile by
   policy* — see :func:`assess_posture`. Only a non-uncertain ``drone``
   detection may be presented as "requires operator verification". Nothing in
   this system is ever "confirmed hostile" automatically, and nothing may
   auto-engage: :class:`EngagementGate` is default-deny and requires an
   explicit operator confirmation, which is written to an audit log.

3. **Uploads and checkpoints are untrusted files.** Uploads are size- and
   extension-checked and stored via :func:`safe_temp_upload` (correct suffix,
   guaranteed cleanup). Checkpoint files are fingerprinted (SHA-256) before
   loading so operators can verify exactly which weights produced a result;
   note that ``.pt`` loading still uses PyTorch pickle deserialization via
   Ultralytics — only load checkpoints you trust (see SECURITY.md).

4. **Values embedded in generated HTML/JS are encoded, never interpolated.**
   :func:`js_literal` produces a JSON literal, so a label such as
   ``x';alert(1);//`` cannot break out of a JavaScript string context.

This module intentionally depends only on the Python standard library so it
can be unit-tested and reused without torch / ultralytics / streamlit.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Limits (resource-exhaustion guards)
# ---------------------------------------------------------------------------

MAX_IMAGE_UPLOAD_BYTES = 20 * 1024 * 1024          # 20 MiB
MAX_VIDEO_UPLOAD_BYTES = 200 * 1024 * 1024         # 200 MiB
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024      # 2 GiB
MIN_CHECKPOINT_BYTES = 1000
MAX_DETECTIONS_PER_FRAME = 1000
DEFAULT_MAX_VIDEO_FRAMES = 100_000

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}

# Classes the civilian product recognises. Anything else a checkpoint emits
# is *unknown*, never a fabricated label.
KNOWN_INTERNAL_CLASSES = {"drone", "airplane", "bird", "balloon"}
NON_HOSTILE_CLASSES = {"airplane", "bird", "balloon"}

SIM_TARGET_ALLOWLIST = {"Drone", "Aircraft", "Bird", "Balloon", "Warplane"}


class DetectionValidationError(ValueError):
    """Raised when a value crossing the AI boundary violates the contract."""


# ---------------------------------------------------------------------------
# Scalar validation
# ---------------------------------------------------------------------------

def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_confidence(value: Any, *, name: str = "confidence") -> float:
    """Return ``value`` as a float, or raise if it is not a finite number in [0, 1]."""
    if not is_finite_number(value):
        raise DetectionValidationError(f"{name} must be a finite number, got {value!r}")
    v = float(value)
    if not (0.0 <= v <= 1.0):
        raise DetectionValidationError(f"{name} must be within [0.0, 1.0], got {v}")
    return v


def validate_thresholds(detection_threshold: Any, uncertainty_threshold: Any) -> Tuple[float, float]:
    """Validate a (detection, uncertainty) threshold pair.

    Both must be finite and in [0, 1], and the uncertainty threshold must not
    be below the detection threshold (otherwise the 'Uncertain' band is empty
    and low-confidence detections would be presented as certain).
    """
    det = validate_confidence(detection_threshold, name="detection_threshold")
    unc = validate_confidence(uncertainty_threshold, name="uncertainty_threshold")
    if unc < det:
        raise DetectionValidationError(
            f"uncertainty_threshold ({unc}) must be >= detection_threshold ({det})"
        )
    return det, unc


def validate_imgsz(imgsz: Any) -> int:
    if not isinstance(imgsz, int) or isinstance(imgsz, bool):
        raise DetectionValidationError(f"imgsz must be an integer, got {imgsz!r}")
    if imgsz < 32 or imgsz > 4096 or imgsz % 32 != 0:
        raise DetectionValidationError(
            f"imgsz must be a multiple of 32 within [32, 4096], got {imgsz}"
        )
    return imgsz


# ---------------------------------------------------------------------------
# Detection contract
# ---------------------------------------------------------------------------

@dataclass
class SanitizedDetection:
    cls_id: int
    conf: float
    box: List[int]  # [x1, y1, x2, y2], clamped to the frame, x2 > x1, y2 > y1


@dataclass
class SanitizeResult:
    detections: List[SanitizedDetection] = field(default_factory=list)
    dropped: Dict[str, int] = field(default_factory=dict)

    def _drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    @property
    def dropped_total(self) -> int:
        return sum(self.dropped.values())


def _coerce_raw(item: Any) -> Optional[Tuple[Any, Any, Any]]:
    """Accept {'cls'/'cid', 'conf', 'xyxy'/'box'} dicts or (cls, conf, box) tuples."""
    if isinstance(item, dict):
        cls_id = item.get("cls", item.get("cid"))
        conf = item.get("conf", item.get("confidence"))
        box = item.get("xyxy", item.get("box"))
        return cls_id, conf, box
    if isinstance(item, (tuple, list)) and len(item) == 3:
        return item[0], item[1], item[2]
    return None


def sanitize_detections(
    raw_detections: Iterable[Any],
    frame_width: int,
    frame_height: int,
    *,
    max_detections: int = MAX_DETECTIONS_PER_FRAME,
) -> SanitizeResult:
    """Validate raw model detections against the detection contract.

    Dropped (with counted reason, never guessed at):
      * ``malformed``        — item does not carry cls/conf/box at all
      * ``bad_class_id``     — class id is not a non-negative integer
      * ``bad_confidence``   — confidence is NaN/inf or outside [0, 1]
      * ``bad_box``          — box is not 4 finite numbers
      * ``degenerate_box``   — zero/negative area after clamping
      * ``overflow``         — beyond ``max_detections`` for one frame

    Surviving boxes are clamped to the frame bounds.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise DetectionValidationError(
            f"frame dimensions must be positive, got {frame_width}x{frame_height}"
        )
    result = SanitizeResult()
    for item in raw_detections:
        if len(result.detections) >= max_detections:
            result._drop("overflow")
            continue
        coerced = _coerce_raw(item)
        if coerced is None:
            result._drop("malformed")
            continue
        cls_raw, conf_raw, box_raw = coerced

        if not is_finite_number(cls_raw) or int(cls_raw) != cls_raw or int(cls_raw) < 0:
            result._drop("bad_class_id")
            continue
        try:
            conf = validate_confidence(conf_raw)
        except DetectionValidationError:
            result._drop("bad_confidence")
            continue
        if (
            not isinstance(box_raw, (tuple, list))
            or len(box_raw) != 4
            or not all(is_finite_number(v) for v in box_raw)
        ):
            result._drop("bad_box")
            continue
        x1, y1, x2, y2 = (float(v) for v in box_raw)
        # Clamp to frame; a box wholly outside the frame becomes degenerate.
        x1 = min(max(x1, 0.0), float(frame_width))
        y1 = min(max(y1, 0.0), float(frame_height))
        x2 = min(max(x2, 0.0), float(frame_width))
        y2 = min(max(y2, 0.0), float(frame_height))
        if x2 <= x1 or y2 <= y1:
            result._drop("degenerate_box")
            continue
        result.detections.append(
            SanitizedDetection(cls_id=int(cls_raw), conf=conf, box=[int(x1), int(y1), int(x2), int(y2)])
        )
    return result


# ---------------------------------------------------------------------------
# Threat posture policy — classification is not hostility
# ---------------------------------------------------------------------------

class ThreatPosture(str, Enum):
    NON_HOSTILE = "non_hostile"              # civilian classes: airplane, bird, balloon
    REQUIRES_VERIFICATION = "requires_verification"  # confident drone: a human must verify
    UNCERTAIN = "uncertain"                  # detection below the uncertainty threshold
    UNKNOWN = "unknown"                      # unrecognised class: no conclusion is drawn


@dataclass(frozen=True)
class PostureAssessment:
    posture: ThreatPosture
    internal_name: Optional[str]
    confidence: float
    # Hard policy constants: this system never auto-engages and never labels
    # a detection "confirmed hostile" on model output alone.
    auto_engage_allowed: bool = False
    confirmed_hostile: bool = False
    requires_human_confirmation: bool = True


def assess_posture(
    internal_name: Optional[str],
    confidence: Any,
    *,
    uncertainty_threshold: float = 0.40,
) -> PostureAssessment:
    """Map a validated detection to its presentation/engagement posture.

    * Civilian classes (Aircraft/Bird/Balloon) are NON_HOSTILE at *any*
      confidence — a civilian airliner detection must never be rendered as a
      hostile target with an engagement prompt (the previous UI did exactly
      that for every "Aircraft" detection).
    * A drone at or above the uncertainty threshold is
      REQUIRES_VERIFICATION — a prompt for a human to look, not a conclusion.
    * Anything below the uncertainty threshold is UNCERTAIN, and an
      unrecognised class is UNKNOWN; neither may drive any action prompt.
    """
    try:
        conf = validate_confidence(confidence)
    except DetectionValidationError:
        return PostureAssessment(ThreatPosture.UNKNOWN, None, 0.0)
    if internal_name is None or internal_name not in KNOWN_INTERNAL_CLASSES:
        return PostureAssessment(ThreatPosture.UNKNOWN, internal_name, conf)
    if conf < uncertainty_threshold:
        return PostureAssessment(ThreatPosture.UNCERTAIN, internal_name, conf)
    if internal_name in NON_HOSTILE_CLASSES:
        return PostureAssessment(ThreatPosture.NON_HOSTILE, internal_name, conf)
    return PostureAssessment(ThreatPosture.REQUIRES_VERIFICATION, internal_name, conf)


# ---------------------------------------------------------------------------
# Human authorization gate for the engagement *simulation*
# ---------------------------------------------------------------------------

CONFIRMATION_PHRASE = "I VERIFY"


@dataclass(frozen=True)
class AuditRecord:
    timestamp_utc: str
    action: str
    target_label: str
    posture: str
    operator: str
    simulation_only: bool = True

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp_utc": self.timestamp_utc,
                "action": self.action,
                "target_label": self.target_label,
                "posture": self.posture,
                "operator": self.operator,
                "simulation_only": self.simulation_only,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


class EngagementGate:
    """Default-deny gate for launching the engagement simulation.

    Authorization requires ALL of:
      * a posture of REQUIRES_VERIFICATION (civilian/uncertain/unknown
        detections can never pass),
      * a non-empty operator identifier,
      * the exact confirmation phrase typed by the operator.

    Every granted authorization produces an :class:`AuditRecord`. The record
    deliberately contains no image data, no file paths, and no credentials.
    """

    def authorize(
        self,
        *,
        action: str,
        target_label: str,
        posture: PostureAssessment,
        operator: Optional[str],
        confirmation: Optional[str],
    ) -> AuditRecord:
        if posture.posture != ThreatPosture.REQUIRES_VERIFICATION:
            raise PermissionError(
                f"Simulation launch denied: posture is '{posture.posture.value}', "
                "only a verified-class drone detection may proceed, and only after human verification."
            )
        if not operator or not str(operator).strip():
            raise PermissionError("Simulation launch denied: operator identifier is required.")
        if confirmation != CONFIRMATION_PHRASE:
            raise PermissionError(
                f"Simulation launch denied: operator must type the confirmation phrase '{CONFIRMATION_PHRASE}'."
            )
        return AuditRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            action=action,
            target_label=target_label,
            posture=posture.posture.value,
            operator=str(operator).strip(),
        )

    @staticmethod
    def append_audit_log(path: Path, record: AuditRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")


# ---------------------------------------------------------------------------
# Upload & checkpoint file guards
# ---------------------------------------------------------------------------

def validate_upload(filename: str, size_bytes: int, *, kind: str) -> str:
    """Validate an uploaded file's extension and size. Returns the safe suffix."""
    if kind == "image":
        allowed, limit = ALLOWED_IMAGE_SUFFIXES, MAX_IMAGE_UPLOAD_BYTES
    elif kind == "video":
        allowed, limit = ALLOWED_VIDEO_SUFFIXES, MAX_VIDEO_UPLOAD_BYTES
    else:
        raise DetectionValidationError(f"unknown upload kind: {kind!r}")
    suffix = Path(filename or "").suffix.lower()
    if suffix not in allowed:
        raise DetectionValidationError(
            f"unsupported {kind} extension '{suffix or '(none)'}'; allowed: {sorted(allowed)}"
        )
    if size_bytes is None or size_bytes < 0:
        raise DetectionValidationError("upload size is unknown")
    if size_bytes == 0:
        raise DetectionValidationError("upload is empty (0 bytes)")
    if size_bytes > limit:
        raise DetectionValidationError(
            f"upload is {size_bytes} bytes, exceeding the {limit}-byte {kind} limit"
        )
    return suffix


@contextmanager
def safe_temp_upload(data_chunks: Iterable[bytes], suffix: str) -> Iterator[Path]:
    """Write an upload to a temp file with its *real* suffix; always clean up.

    The previous code forced ``suffix='.mp4'`` for every upload (including
    AVI/MOV files) and used ``delete=False`` with no cleanup, leaking files
    and mislabelling containers for downstream decoders.
    """
    if suffix not in ALLOWED_IMAGE_SUFFIXES | ALLOWED_VIDEO_SUFFIXES:
        raise DetectionValidationError(f"unsafe temp suffix: {suffix!r}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = Path(tmp.name)
    try:
        with tmp:
            for chunk in data_chunks:
                tmp.write(chunk)
        yield path
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def checkpoint_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_checkpoint_file(path: Path) -> str:
    """Validate a checkpoint before any deserialization; return its SHA-256.

    Checks existence, ``.pt`` extension, and a sane size window. The hash lets
    operators pin/verify the exact weights in use. This does NOT make pickle
    loading safe — only load checkpoints from trusted sources.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {p}")
    if p.suffix.lower() != ".pt":
        raise DetectionValidationError(f"checkpoint must be a .pt file, got '{p.suffix}'")
    size = p.stat().st_size
    if size < MIN_CHECKPOINT_BYTES:
        raise DetectionValidationError(f"checkpoint is suspiciously small ({size} bytes): {p.name}")
    if size > MAX_CHECKPOINT_BYTES:
        raise DetectionValidationError(f"checkpoint exceeds the {MAX_CHECKPOINT_BYTES}-byte limit: {p.name}")
    return checkpoint_sha256(p)


# ---------------------------------------------------------------------------
# Safe embedding into generated HTML/JS
# ---------------------------------------------------------------------------

def js_literal(value: Any) -> str:
    """Encode a value as a JSON literal safe to embed in a <script> block."""
    return json.dumps(value, ensure_ascii=True).replace("</", "<\\/")


def validate_sim_target(target_type: Any) -> str:
    """Allowlist the simulator's target label; anything else becomes 'Drone'."""
    if isinstance(target_type, str) and target_type in SIM_TARGET_ALLOWLIST:
        return target_type
    return "Drone"


def clamp_finite(value: Any, low: float, high: float, default: float) -> float:
    if not is_finite_number(value):
        return default
    return min(max(float(value), low), high)
