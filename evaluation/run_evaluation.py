"""AeroWatch AI — contract-level AI evaluation runner.

Runs evaluation/cases.json against the *current* code paths that consume
model output:

    raw detection -> ai_safety.sanitize_detections
                  -> ModelManager.map_detection_label
                  -> ai_safety.assess_posture
                  -> ai_safety.EngagementGate
                  -> interceptor_sim HTML embedding

Usage:
    python evaluation/run_evaluation.py [--cases PATH] [--out PATH]

Exit code 0 iff every case passes. A JSON report is written to
evaluation/results/latest.json by default.

What this measures: correctness of the output contract, groundedness (no
fabricated labels, no automatic hostility), failure handling for malformed
model output, injection resistance, and authorization boundaries.

What this does NOT measure: detector accuracy (mAP/precision/recall). That
requires labelled imagery and is covered by tests/test_model_evaluation.py;
the only detector metrics with repository evidence live in
reports/validation/summary.json (30 images: precision 0.805, recall 0.868,
estimated mAP50 0.699 — see docs/AI_ENGINEERING.md for the claim audit).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_safety import (  # noqa: E402
    EngagementGate,
    assess_posture,
    js_literal,
    sanitize_detections,
)

_SENTINELS = {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": float("-inf")}


def _desentinel(value: Any) -> Any:
    if isinstance(value, str) and value in _SENTINELS:
        return _SENTINELS[value]
    if isinstance(value, list):
        return [_desentinel(v) for v in value]
    if isinstance(value, dict):
        return {k: _desentinel(v) for k, v in value.items()}
    return value


def _fake_metadata_manager():
    from model_manager import CheckpointMetadata, ModelManager, ModelType

    mm = ModelManager()
    mm._metadata = CheckpointMetadata(
        path=Path("evaluation-fake.pt"),
        raw_names={0: "drone", 1: "aircraft", 2: "bird", 3: "balloon"},
        model_type=ModelType.CUSTOM_CIVILIAN,
        device="cpu",
        is_half=False,
    )
    return mm


def run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    kind = case["kind"]
    expect = case.get("expect", {})
    failures: List[str] = []
    observed: Dict[str, Any] = {}

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    if kind == "sanitize":
        frame = case.get("frame", [640, 480])
        res = sanitize_detections(_desentinel(case["raw_list"]), frame[0], frame[1])
        observed = {"kept": len(res.detections), "dropped": res.dropped}
        if "kept" in expect:
            check(len(res.detections) == expect["kept"], f"kept={len(res.detections)} != {expect['kept']}")
        if "dropped_total" in expect:
            check(res.dropped_total == expect["dropped_total"], f"dropped_total={res.dropped_total}")
        if "drop_reason" in expect:
            check(expect["drop_reason"] in res.dropped, f"missing drop reason {expect['drop_reason']}")
        if "first_box" in expect and res.detections:
            check(res.detections[0].box == expect["first_box"], f"first_box={res.detections[0].box}")

    elif kind == "sanitize_repeat":
        raw = _desentinel(case["raw"])
        res = sanitize_detections([dict(raw) for _ in range(case["repeat"])], 640, 480)
        observed = {"kept": len(res.detections), "dropped": res.dropped}
        if "kept" in expect:
            check(len(res.detections) == expect["kept"], f"kept={len(res.detections)} != {expect['kept']}")
        if "drop_reason" in expect:
            check(expect["drop_reason"] in res.dropped, "missing overflow drop reason")

    elif kind == "pipeline":
        raw = _desentinel(case["raw"])
        res = sanitize_detections([raw], 640, 480)
        mm = _fake_metadata_manager()
        internal = display = None
        posture = None
        if res.detections:
            det = res.detections[0]
            internal, display, _ = mm.map_detection_label(det.cls_id, det.conf, 0.25, 0.40)
            posture = assess_posture(internal, det.conf, uncertainty_threshold=0.40).posture.value
        observed = {"kept": len(res.detections), "internal": internal, "display": display, "posture": posture}
        if "kept" in expect:
            check(len(res.detections) == expect["kept"], f"kept={len(res.detections)}")
        if "internal" in expect:
            check(internal == expect["internal"], f"internal={internal} != {expect['internal']}")
        if "posture" in expect:
            check(posture == expect["posture"], f"posture={posture} != {expect['posture']}")

    elif kind == "posture":
        name, conf = case["posture_of"]
        a = assess_posture(name, conf)
        observed = {
            "posture": a.posture.value,
            "confirmed_hostile": a.confirmed_hostile,
            "auto_engage_allowed": a.auto_engage_allowed,
        }
        for key, val in expect.items():
            check(observed.get(key) == val, f"{key}={observed.get(key)} != {val}")

    elif kind == "gate":
        name, conf = case["posture_of"]
        posture = assess_posture(name, conf)
        authorized = False
        record = None
        try:
            record = EngagementGate().authorize(
                action="open_interceptor_simulation",
                target_label=name,
                posture=posture,
                operator=case.get("operator"),
                confirmation=case.get("confirmation"),
            )
            authorized = True
        except PermissionError:
            authorized = False
        observed = {"authorized": authorized}
        check(authorized == expect["authorized"], f"authorized={authorized} != {expect['authorized']}")
        if expect.get("audit_simulation_only") and record is not None:
            check(record.simulation_only is True, "audit record is not marked simulation_only")

    elif kind == "sim_injection":
        from interceptor_sim import get_interceptor_sim_html

        html = get_interceptor_sim_html(target_type=case["target_type"])
        observed = {"html_len": len(html)}
        if "html_must_not_contain" in expect:
            check(expect["html_must_not_contain"] not in html, "injection payload present in HTML")
        if "html_must_contain" in expect:
            check(expect["html_must_contain"] in html, "expected safe literal missing from HTML")

    elif kind == "js_literal":
        lit = js_literal(case["value"])
        observed = {"literal": lit}
        if "literal_must_not_contain" in expect:
            check(expect["literal_must_not_contain"] not in lit, "unsafe literal produced")

    else:
        failures.append(f"unknown case kind: {kind}")

    return {
        "id": case["id"],
        "category": case["category"],
        "passed": not failures,
        "failures": failures,
        "observed": observed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AeroWatch AI contract evaluation")
    parser.add_argument("--cases", default=str(Path(__file__).parent / "cases.json"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "results" / "latest.json"))
    args = parser.parse_args()

    suite = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    results = [run_case(c) for c in suite["cases"]]

    by_category: Dict[str, Dict[str, int]] = {}
    for r in results:
        bucket = by_category.setdefault(r["category"], {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(r["passed"])

    passed = sum(r["passed"] for r in results)
    report = {
        "suite": suite.get("suite"),
        "total": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "by_category": by_category,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['category']:20s} {r['id']}" + (f" — {'; '.join(r['failures'])}" if r["failures"] else ""))
    print(f"\n{passed}/{len(results)} cases passed ({report['pass_rate']:.0%}) -> {out_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
