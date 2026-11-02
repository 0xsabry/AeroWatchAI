"""Tests for the ai_safety contract layer and its pipeline integrations.

Coverage map (evaluation categories):
- normal requests ............ valid detections pass through unchanged
- malformed model output ..... NaN/inf confidence, inverted boxes, bad ids
- ambiguous / uncertain ...... sub-uncertainty detections, unknown classes
- prompt/JS injection ........ hostile labels embedded in generated HTML
- privacy / authorization .... engagement gate is default-deny, audit log
                               contains no image data or credentials
- provider failure ........... corrupt/missing checkpoints rejected loudly
- resource exhaustion ........ upload limits, per-frame detection cap,
                               video frame limit flag
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ai_safety import (
    CONFIRMATION_PHRASE,
    DetectionValidationError,
    EngagementGate,
    ThreatPosture,
    assess_posture,
    checkpoint_sha256,
    dedupe_canonical,
    js_literal,
    safe_temp_upload,
    sanitize_detections,
    validate_checkpoint_file,
    validate_confidence,
    validate_imgsz,
    validate_thresholds,
    validate_upload,
    validate_sim_target,
)

# ---------------------------------------------------------------------------
# Confidence / threshold contract
# ---------------------------------------------------------------------------

class TestConfidenceContract:
    def test_valid_confidence_passes(self):
        assert validate_confidence(0.5) == 0.5
        assert validate_confidence(0) == 0.0
        assert validate_confidence(1) == 1.0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.5, "0.9", None, True])
    def test_invalid_confidence_rejected(self, bad):
        with pytest.raises(DetectionValidationError):
            validate_confidence(bad)

    def test_threshold_pair_must_be_ordered(self):
        assert validate_thresholds(0.25, 0.40) == (0.25, 0.40)
        with pytest.raises(DetectionValidationError):
            validate_thresholds(0.50, 0.40)  # empty 'Uncertain' band

    def test_imgsz_contract(self):
        assert validate_imgsz(640) == 640
        for bad in (0, -32, 641, 8192, "640", 640.0):
            with pytest.raises(DetectionValidationError):
                validate_imgsz(bad)


# ---------------------------------------------------------------------------
# Detection sanitization
# ---------------------------------------------------------------------------

class TestSanitizeDetections:
    def test_normal_detection_passes_unchanged(self):
        res = sanitize_detections([{"cls": 0, "conf": 0.9, "xyxy": [10, 10, 50, 60]}], 100, 100)
        assert len(res.detections) == 1
        assert res.detections[0].box == [10, 10, 50, 60]
        assert res.dropped == {}

    def test_nan_confidence_dropped_not_displayed(self):
        # Regression: NaN previously passed `conf < threshold` checks.
        res = sanitize_detections([{"cls": 0, "conf": float("nan"), "xyxy": [0, 0, 10, 10]}], 100, 100)
        assert res.detections == []
        assert res.dropped == {"bad_confidence": 1}

    def test_out_of_range_confidence_dropped(self):
        res = sanitize_detections([{"cls": 0, "conf": 1.7, "xyxy": [0, 0, 10, 10]}], 100, 100)
        assert res.dropped == {"bad_confidence": 1}

    def test_nan_box_dropped_without_crashing(self):
        # Regression: int(NaN) previously raised and killed frame processing.
        res = sanitize_detections(
            [{"cls": 0, "conf": 0.9, "xyxy": [0, float("nan"), 10, 10]}], 100, 100
        )
        assert res.dropped == {"bad_box": 1}

    def test_inverted_and_outside_boxes(self):
        res = sanitize_detections(
            [
                {"cls": 0, "conf": 0.9, "xyxy": [50, 50, 10, 10]},      # inverted
                {"cls": 0, "conf": 0.9, "xyxy": [200, 200, 300, 300]},  # wholly outside
                {"cls": 0, "conf": 0.9, "xyxy": [-20, -20, 30, 30]},    # partially outside -> clamped
            ],
            100, 100,
        )
        assert [d.box for d in res.detections] == [[0, 0, 30, 30]]
        assert res.dropped == {"degenerate_box": 2}

    def test_bad_class_ids_dropped(self):
        res = sanitize_detections(
            [
                {"cls": -1, "conf": 0.9, "xyxy": [0, 0, 10, 10]},
                {"cls": 1.5, "conf": 0.9, "xyxy": [0, 0, 10, 10]},
                {"conf": 0.9, "xyxy": [0, 0, 10, 10]},
            ],
            100, 100,
        )
        assert res.detections == []
        assert res.dropped.get("bad_class_id") == 3

    def test_per_frame_cap_enforced(self):
        items = [{"cls": 0, "conf": 0.9, "xyxy": [0, 0, 10, 10]}] * 5
        res = sanitize_detections(items, 100, 100, max_detections=3)
        assert len(res.detections) == 3
        assert res.dropped == {"overflow": 2}

    def test_invalid_frame_dimensions_raise(self):
        with pytest.raises(DetectionValidationError):
            sanitize_detections([], 0, 100)


# ---------------------------------------------------------------------------
# Threat posture policy
# ---------------------------------------------------------------------------

class TestThreatPosture:
    def test_civilian_aircraft_is_never_hostile(self):
        # Regression: the UI previously rendered every Aircraft detection as
        # "Hostile … confirmed" with an ENGAGE button.
        a = assess_posture("airplane", 0.99)
        assert a.posture == ThreatPosture.NON_HOSTILE
        assert a.confirmed_hostile is False
        assert a.auto_engage_allowed is False

    @pytest.mark.parametrize("cls", ["bird", "balloon"])
    def test_other_civilian_classes_non_hostile(self, cls):
        assert assess_posture(cls, 0.95).posture == ThreatPosture.NON_HOSTILE

    def test_confident_drone_requires_verification_only(self):
        a = assess_posture("drone", 0.9)
        assert a.posture == ThreatPosture.REQUIRES_VERIFICATION
        assert a.confirmed_hostile is False
        assert a.auto_engage_allowed is False

    def test_low_confidence_drone_is_uncertain(self):
        assert assess_posture("drone", 0.30).posture == ThreatPosture.UNCERTAIN

    def test_unknown_class_and_nan_confidence_are_unknown(self):
        assert assess_posture("kamikaze_drone", 0.99).posture == ThreatPosture.UNKNOWN
        assert assess_posture("drone", float("nan")).posture == ThreatPosture.UNKNOWN
        assert assess_posture(None, 0.9).posture == ThreatPosture.UNKNOWN


# ---------------------------------------------------------------------------
# Engagement gate (authorization boundary)
# ---------------------------------------------------------------------------

class TestEngagementGate:
    def _drone_posture(self):
        return assess_posture("drone", 0.9)

    def test_default_deny_without_confirmation(self):
        with pytest.raises(PermissionError):
            EngagementGate().authorize(
                action="open_interceptor_simulation",
                target_label="Drone",
                posture=self._drone_posture(),
                operator="op-1",
                confirmation=None,
            )

    def test_deny_without_operator(self):
        with pytest.raises(PermissionError):
            EngagementGate().authorize(
                action="open_interceptor_simulation",
                target_label="Drone",
                posture=self._drone_posture(),
                operator="  ",
                confirmation=CONFIRMATION_PHRASE,
            )

    def test_civilian_posture_can_never_be_authorized(self):
        with pytest.raises(PermissionError):
            EngagementGate().authorize(
                action="open_interceptor_simulation",
                target_label="Aircraft",
                posture=assess_posture("airplane", 0.99),
                operator="op-1",
                confirmation=CONFIRMATION_PHRASE,
            )

    def test_authorized_launch_produces_minimal_audit_record(self, tmp_path):
        gate = EngagementGate()
        record = gate.authorize(
            action="open_interceptor_simulation",
            target_label="Drone",
            posture=self._drone_posture(),
            operator="op-1",
            confirmation=CONFIRMATION_PHRASE,
        )
        log = tmp_path / "audit.jsonl"
        gate.append_audit_log(log, record)
        entry = json.loads(log.read_text(encoding="utf-8").strip())
        assert entry["operator"] == "op-1"
        assert entry["simulation_only"] is True
        # Privacy: the audit record must not carry media or secrets.
        blob = json.dumps(entry).lower()
        for forbidden in ("image", "base64", "password", "token", ".jpg", ".mp4", "path"):
            assert forbidden not in blob


# ---------------------------------------------------------------------------
# Upload / checkpoint guards
# ---------------------------------------------------------------------------

class TestFileGuards:
    def test_upload_extension_and_size(self):
        assert validate_upload("clip.AVI", 1024, kind="video") == ".avi"
        with pytest.raises(DetectionValidationError):
            validate_upload("clip.exe", 1024, kind="video")
        with pytest.raises(DetectionValidationError):
            validate_upload("clip.mp4", 0, kind="video")
        with pytest.raises(DetectionValidationError):
            validate_upload("photo.png", 21 * 1024 * 1024, kind="image")

    def test_safe_temp_upload_uses_real_suffix_and_cleans_up(self):
        with safe_temp_upload([b"abc", b"def"], ".avi") as p:
            assert p.suffix == ".avi"
            assert p.read_bytes() == b"abcdef"
            kept = p
        assert not kept.exists()

    def test_safe_temp_upload_rejects_executable_suffix(self):
        with pytest.raises(DetectionValidationError):
            with safe_temp_upload([b"x"], ".exe"):
                pass

    def test_checkpoint_validation_and_fingerprint(self, tmp_path):
        good = tmp_path / "model.pt"
        good.write_bytes(b"x" * 2048)
        digest = validate_checkpoint_file(good)
        assert digest == checkpoint_sha256(good)
        assert len(digest) == 64

        with pytest.raises(FileNotFoundError):
            validate_checkpoint_file(tmp_path / "missing.pt")
        wrong_ext = tmp_path / "model.bin"
        wrong_ext.write_bytes(b"x" * 2048)
        with pytest.raises(DetectionValidationError):
            validate_checkpoint_file(wrong_ext)
        tiny = tmp_path / "tiny.pt"
        tiny.write_bytes(b"x" * 10)
        with pytest.raises(DetectionValidationError):
            validate_checkpoint_file(tiny)


# ---------------------------------------------------------------------------
# HTML/JS embedding (injection resistance)
# ---------------------------------------------------------------------------

class TestJsEmbedding:
    def test_js_literal_escapes_script_breakout(self):
        payload = "x';alert(1);//"
        lit = js_literal(payload)
        # A valid JSON string literal that round-trips to the exact payload:
        # embedded in JS it is one string value, never executable code.
        assert json.loads(lit) == payload
        assert "</script>" not in js_literal("</script><script>alert(1)</script>")

    def test_sim_target_allowlist(self):
        assert validate_sim_target("Drone") == "Drone"
        assert validate_sim_target("x';alert(1);//") == "Drone"
        assert validate_sim_target(None) == "Drone"

    def test_interceptor_sim_neutralizes_injection(self):
        from interceptor_sim import get_interceptor_sim_html

        html = get_interceptor_sim_html(target_type="x';alert(1);//")
        assert "alert(1);//" not in html
        assert 'type: "Drone"' in html

    def test_interceptor_sim_sanitizes_non_finite_numbers(self):
        from interceptor_sim import get_interceptor_sim_html

        html = get_interceptor_sim_html(target_type="Drone", evasion_g=float("nan"))
        assert "evasionG: 6.0" in html
        html_inf = get_interceptor_sim_html(target_type="Drone", evasion_g=float("inf"))
        assert "evasionG: Infinity" not in html_inf


# ---------------------------------------------------------------------------
# ModelManager integration with the contract
# ---------------------------------------------------------------------------

class TestModelManagerContract:
    def test_map_label_rejects_nan_and_out_of_range_confidence(self):
        from model_manager import ModelManager

        mm = ModelManager()
        assert mm.map_detection_label(0, float("nan")) == (None, None, None)
        assert mm.map_detection_label(0, 1.5) == (None, None, None)
        assert mm.map_detection_label(-3, 0.9) == (None, None, None)

    def test_undeclared_class_id_is_unverified_not_fabricated(self):
        from model_manager import ModelManager

        pt = Path("weights/aerowatch_v7_balanced.pt")
        if not pt.exists():
            pytest.skip("weights not available")
        mm = ModelManager()
        mm.load_model(pt, device="cpu", enable_half=False)
        internal, display, _ = mm.map_detection_label(999, 0.95)
        assert internal == "unknown"
        assert "Unverified" in display

    def test_unrecognised_declared_class_is_unverified(self):
        from model_manager import CheckpointMetadata, ModelManager, ModelType

        mm = ModelManager()
        mm._metadata = CheckpointMetadata(
            path=Path("fake.pt"),
            raw_names={0: "drone", 1: "missile"},
            model_type=ModelType.CUSTOM_CIVILIAN,
            device="cpu",
            is_half=False,
        )
        internal, display, _ = mm.map_detection_label(1, 0.95)
        assert internal == "unknown"
        assert "Missile" not in display

    def test_compound_drone_name_normalizes_to_drone(self):
        # Established project normalization (see AUDIT_REPORT.md): compound
        # class names containing a canonical class map to that class, e.g.
        # "military_jet" -> Aircraft and "kamikaze_drone" -> Drone.
        from model_manager import CheckpointMetadata, ModelManager, ModelType

        mm = ModelManager()
        mm._metadata = CheckpointMetadata(
            path=Path("fake.pt"),
            raw_names={0: "kamikaze_drone"},
            model_type=ModelType.CUSTOM_CIVILIAN,
            device="cpu",
            is_half=False,
        )
        internal, display, _ = mm.map_detection_label(0, 0.95)
        assert (internal, display) == ("drone", "Drone")

    def test_filter_drops_nan_box_instead_of_crashing(self):
        import numpy as np

        from model_manager import ModelManager

        class FakeTensor:
            def __init__(self, v):
                self._v = v

            def item(self):
                return self._v

            def tolist(self):
                return self._v

        class FakeBox:
            def __init__(self, cls, conf, xyxy):
                self.cls = FakeTensor(cls)
                self.conf = FakeTensor(conf)
                self.xyxy = [FakeTensor(xyxy)]

        mm = ModelManager()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [
            FakeBox(0, float("nan"), [0, 0, 10, 10]),
            FakeBox(0, 0.9, [0, float("nan"), 10, 10]),
        ]
        # No metadata loaded: valid boxes map to Unknown (Unverified); both
        # malformed boxes here must be dropped, not raise.
        assert mm.filter_airspace_detections(frame, boxes, 0.25, 0.40) == []


# ---------------------------------------------------------------------------
# Canonical duplicate suppression (Phase 2, measured in docs/CV_QUALITY.md)
# ---------------------------------------------------------------------------

def _det(name, conf, box):
    return {"internal_name": name, "conf": conf, "box": box}


def test_dedupe_suppresses_cross_class_duplicate_keeps_highest_conf():
    # The v7 checkpoint declares aircraft + military_jet, both normalised
    # to "airplane": one physical aircraft, two near-identical boxes.
    dets = [
        _det("airplane", 0.33, [52, 835, 832, 1191]),
        _det("airplane", 0.43, [59, 837, 833, 1191]),
    ]
    kept, suppressed = dedupe_canonical(dets)
    assert suppressed == 1
    assert len(kept) == 1 and kept[0]["conf"] == 0.43


def test_dedupe_keeps_distinct_objects_and_other_classes():
    dets = [
        _det("airplane", 0.9, [0, 0, 100, 100]),
        _det("airplane", 0.8, [300, 300, 400, 400]),   # distinct aircraft
        _det("drone", 0.85, [5, 5, 95, 95]),           # overlap, other class
    ]
    kept, suppressed = dedupe_canonical(dets)
    assert suppressed == 0 and len(kept) == 3


def test_dedupe_partial_overlap_below_threshold_is_kept():
    dets = [
        _det("airplane", 0.9, [0, 0, 100, 100]),
        _det("airplane", 0.8, [60, 0, 160, 100]),  # IoU 0.25 < 0.7
    ]
    kept, suppressed = dedupe_canonical(dets)
    assert suppressed == 0 and len(kept) == 2
