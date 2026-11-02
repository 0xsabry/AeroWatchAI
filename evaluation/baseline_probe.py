"""Baseline probe: measure the *pre-fix* behaviour of a checkout.

Points at a directory containing the original (pre-ai_safety) copies of
model_manager.py and interceptor_sim.py and checks four legacy behaviours
that the ai_safety work changed:

  1. NaN confidence handling in map_detection_label
     (legacy: NaN passes every threshold comparison -> labelled as a target;
      current: rejected as invalid model output)
  2. Undeclared/unknown class handling
     (legacy: the raw class name is passed through as a trusted label;
      current: surfaced as "Unknown (Unverified)")
  3. Civilian-as-hostile rule from the legacy app.py
     (legacy: display labels ["Drone", "Aircraft"] were treated as hostile and
      offered a one-click ENGAGE action; current: posture policy in ai_safety)
  4. Simulator HTML injection
     (legacy: target label interpolated raw into a JS string literal;
      current: allowlist + JSON literal encoding)

Usage:
    python evaluation/baseline_probe.py /path/to/original/files

The probe imports the target directory's modules in isolation. It needs the
same third-party deps as the project (torch, ultralytics, numpy).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1]).resolve()
    mm_mod = _load("baseline_model_manager", target / "model_manager.py")
    sim_mod = _load("baseline_interceptor_sim", target / "interceptor_sim.py")

    mm = mm_mod.ModelManager()
    nan_result = mm.map_detection_label(0, float("nan"))
    unknown_result = mm.map_detection_label(999, 0.95)

    # Legacy app.py rule (pre-fix): hostiles = labels in ["Drone", "Aircraft"]
    legacy_hostile_labels = {"Drone", "Aircraft"}
    legacy_aircraft_hostile = "Aircraft" in legacy_hostile_labels

    payload = "x';alert(1);//"
    html = sim_mod.get_interceptor_sim_html(target_type=payload)
    injection_present = payload in html

    findings = {
        "target_dir": str(target),
        "nan_confidence_labelled_as_target": nan_result[1] is not None,
        "nan_confidence_result": list(nan_result),
        "undeclared_class_result": list(unknown_result),
        "undeclared_class_passed_through": unknown_result[1] not in (None, "Unknown (Unverified)"),
        "legacy_rule_marks_civilian_aircraft_hostile": legacy_aircraft_hostile,
        "sim_injection_payload_present_verbatim": injection_present,
    }
    print(json.dumps(findings, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
