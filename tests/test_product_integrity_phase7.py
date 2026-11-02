"""Phase 7 product-integrity guards.

The public identity of the project (README, structured metadata,
citation file, and every user-facing string in the app) must agree with
what the code actually does:

- no guaranteed/certified/absolute performance claims,
- no false capability claims (notably persistent tracking),
- measured figures only, with their evidence level stated,
- AI-generated demo footage disclosed as synthetic wherever offered,
- the interceptor always presented as a simulation game.

Historical audit documents under docs/ and dated 2026-10-12 reports are
out of scope for the string scans: they either record the audits
themselves or now carry superseded banners. Negations in README.md
("no ByteTrack", "figures ... are withdrawn") are the honest form and
are handled by banning claim phrasings, not bare words.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

USER_FACING_CODE = [
    "app.py",
    "interceptor_sim.py",
    "video_interceptor_testbench.py",
    "video_processor.py",
    "predict_image.py",
]

BANNED_CLAIM_PHRASES = [
    "100% Pk",
    "100.0% Pk",
    "100% Kill Probability",
    "Guaranteed Neutralization",
    "guaranteed neutralization",
    "Combat-Certified",
    "CERTIFIED COMBAT",
    "100% Lethal",
    "KILL PROBABILITY (Pk)",
    "Hypersonic Solid Motor",
    "Guarantees 100%",
    "Zero False Alarms",
    "zero false-alarm",
    "inescapable pincer",
    "Military-grade",
    "military-grade autonomous",
]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_no_banned_claims_in_user_facing_code():
    for rel in USER_FACING_CODE:
        text = _read(rel)
        for phrase in BANNED_CLAIM_PHRASES:
            assert phrase not in text, f"{rel} contains claim: {phrase!r}"


def test_no_banned_claims_in_public_docs():
    for rel in ["README.md", "README_AR.md", "CITATION.cff"]:
        text = _read(rel)
        for phrase in BANNED_CLAIM_PHRASES:
            assert phrase not in text, f"{rel} contains claim: {phrase!r}"
        assert "96.7% Drone Precision" not in text
        assert "96.7% precision" not in text
        assert "$P_k = 1.00$" not in text


def test_readme_reports_measured_independent_results():
    readme = _read("README.md")
    # The Phase 5 independent-test figures are the headline numbers.
    for token in ("0.497", "0.363", "0.151", "independent"):
        assert token in readme, f"README missing measured figure: {token}"
    # The withdrawn marketing figures may only appear as withdrawn.
    assert "are **withdrawn**" in readme


def test_readme_json_ld_is_valid_and_truthful():
    readme = _read("README.md")
    match = re.search(
        r'<script type="application/ld\+json">(.*?)</script>',
        readme,
        re.DOTALL,
    )
    assert match, "README has no JSON-LD block"
    data = json.loads(match.group(1))
    assert data["name"] == "AeroWatch AI"
    description = data["description"].lower()
    assert "simulation game" in description
    assert "does not control" in description
    for banned in ("hypersonic", "fire control", "military-grade",
                   "100%", "guaranteed", "certified"):
        assert banned not in description, f"JSON-LD claims: {banned}"


def test_no_false_tracking_claims():
    # The application code must not claim ByteTrack/Kalman tracking.
    for rel in ["app.py", "video_processor.py"]:
        text = _read(rel)
        assert "ByteTrack" not in text, f"{rel} claims ByteTrack"
        assert "Kalman" not in text, f"{rel} claims Kalman filtering"
    # CITATION keywords must not advertise tracking that the app lacks.
    cff = _read("CITATION.cff").lower()
    assert "bytetrack" not in cff
    assert "kalman" not in cff
    # README must state the truth: no persistent tracking in the app,
    # and must not present tracking as a feature anywhere.
    readme = _read("README.md")
    assert "does **not** implement" in readme
    assert "persistent object tracking" in readme
    assert "ByteTrack +" not in readme
    assert "ByteTrack persistent" not in readme


def test_ai_generated_demo_videos_disclosed():
    app = _read("app.py")
    # Both Gemini-generated presets are labelled in the picker itself...
    assert "AI-Generated Demo 01" in app
    assert "AI-Generated Demo 02" in app
    assert "synthetic footage — not a real recording" in app
    # ...and selecting one shows an explicit disclosure banner.
    assert "AI-generated demonstration footage" in app
    assert "not** a real camera" in app or "not a real camera" in app


def test_simulation_framing_present_in_app():
    app = _read("app.py")
    assert "Simulation only" in app
    assert "not a real-world result or certification" in app
    assert "SIMULATION SUCCESS RATE" in app


def test_legacy_reports_carry_superseded_banners():
    for rel in ["AUDIT_REPORT.md", "BENCHMARK_REPORT.md",
                "VIDEO_AND_ACCURACY_REPORT.md"]:
        assert "superseded" in _read(rel).lower(), f"{rel} lacks banner"
