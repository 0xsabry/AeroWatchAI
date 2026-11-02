# AeroWatch AI — AI Engineering Notes

What the AI in this repository actually is, what was verified, what was
broken, and what changed in the `feat/ai-safety-reliability` work.

## 1. The actual AI architecture (as found)

There is **no LLM, no prompt layer, and no external AI API** in this
codebase. The "AI" is a single computer-vision pipeline:

```
image / video / camera frame
  -> Ultralytics YOLO checkpoint (weights/*.pt, custom 4-class civilian
     vocabulary: drone, aircraft, bird, balloon; or a COCO 80-class
     baseline restricted to airplane/bird)
  -> ModelManager.map_detection_label      (class normalisation + thresholds)
  -> ModelManager.filter_airspace_detections (geometric filters + COCO
     person-overlap suppression)
  -> VideoProcessor / Streamlit app        (rendering, metrics, UI)
  -> interceptor_sim.py                    (a self-contained HTML5 canvas
     *simulation game*, not connected to any real hardware or effector)
```

Components: `model_manager.py` (loading, normalisation, filtering),
`video_processor.py` (profiled video inference), `camera_manager.py`
(threaded capture), `app.py` (Streamlit UI, 2,300+ lines), `predict_image.py`
(CLI). The interceptor "fire control" is a browser canvas simulation whose
physics are in pixels/second; it has no path to any real system.

## 2. Claim audit (README vs repository evidence)

The README's headline numbers are **not supported** by the repository's own
validation artifacts (`reports/validation/summary.json`, generated
2026-10-12 on an RTX 3050 Laptop, 30 test images / 38 ground-truth targets):

| Claim (README) | Repository evidence |
| --- | --- |
| 84.6% mAP@50 | `map50_estimated: 0.699` (estimated, n=38 targets) |
| 96.7% drone precision | Per-class Drone precision 0.933 (14 TP / 1 FP) |
| 3.0 ms inference latency | `benchmark.csv`: mean inference 11.8–15.8 ms (FP16, RTX 3050) |
| 100% Pk / "combat certified" | No artifact measures Pk; the "certification" is the canvas simulation's own scenario run |
| "Zero false alarms" | Validation run recorded 8 false positives (Balloon precision 0.50) |

These are small-sample, single-run numbers even where artifacts exist. Treat
every performance figure as *unverified* until reproduced with the evaluation
harness below on a documented dataset. A "Verification status" note has been
added to the README pointing here.

## 3. Defects found (all reproduced before fixing)

| # | Severity | Defect | Evidence |
| --- | --- | --- | --- |
| 1 | **Critical (safety)** | The image UI labelled every `Aircraft` detection — the class that includes civilian airliners — as *"Hostile … confirmed"* with a one-click **ENGAGE** button (`app.py`, legacy rule `["Drone", "Aircraft"]`) | `evaluation/baseline_probe.py`: `legacy_rule_marks_civilian_aircraft_hostile: true` |
| 2 | **High** | `NaN` confidence passed all threshold checks (NaN comparisons are false) and was returned/displayed as a real target | Baseline probe: `map_detection_label(0, NaN)` → `("cls_0", "Class 0")`; filter returned the detection with `conf: nan` |
| 3 | **High** | `NaN` box coordinate crashed `filter_airspace_detections` via `int(NaN)` → `ValueError: cannot convert float NaN to integer`, killing frame processing | Reproduced against baseline code |
| 4 | **High (security)** | `interceptor_sim.py` interpolated the target label raw into a JavaScript string (`type: '{target_type}'`): a label containing `'` injects arbitrary script into the C2 page | Baseline HTML contains `type: 'x';alert(1);//',` verbatim |
| 5 | **Medium** | Undeclared/unknown class ids were passed through as trusted labels (`cls_999` → "Class 999"; unrecognised names title-cased) — fabricated labels downstream | Baseline probe: `undeclared_class_passed_through: true` |
| 6 | **Medium** | Video uploads: forced `.mp4` suffix for AVI/MOV files, no size limit, whole file read into memory, temp files never deleted (`delete=False`) | `app.py` legacy upload blocks (×2) |
| 7 | **Medium** | No resource limits in `VideoProcessor`: unbounded frame loop, `width/height = 0` from a corrupt header caused division by zero in coverage maths | `video_processor.py` legacy code path |
| 8 | **Low (perf)** | `load_model(device="auto")` loaded and ran a second throwaway copy of the model as a "probe", then always chose CPU anyway; `predict_image` paid a full extra model load via `introspect_checkpoint` + `load_model` | `model_manager.py` legacy code |
| 9 | **Low** | `predict_image.py` accepted out-of-range `--conf`, non-multiple `--imgsz`, and a path-traversing `--name` (`../../`) | CLI argument definitions |
| 9b | **Medium** | `evaluate_videos.py` crashed with an unhandled `KeyError` for any class id ≥ 5 (e.g. an 80-class COCO checkpoint), trusted raw boxes like the other paths, and silently swallowed H.264 re-encode failures, presenting mp4v output as browser-ready | `evaluate_videos.py` legacy code (found by the security audit round) |
| 9c | **Medium** | Model/file-derived strings (`target_class`, checkpoint filename) were interpolated unescaped into `unsafe_allow_html` blocks; ffmpeg subprocesses had no timeout; the video temp path was predictable (`temp_<name>`) | `app.py` / `video_processor.py` legacy code |
| 10 | **Info** | Checkpoints are PyTorch pickle (`.pt`) loaded via Ultralytics with no integrity check — a malicious `.pt` is code execution. Mitigation is procedural (trusted sources + hash pinning), not eliminable while using `.pt` | `model_manager.py` load path |

## 4. What changed

New module **`ai_safety.py`** (stdlib-only) — the single contract boundary
for model output and consequential UI actions:

- `sanitize_detections()` — validates every raw detection (finite confidence
  in [0, 1], finite box, positive area, clamped to frame, per-frame cap).
  Malformed items are dropped with counted reasons, never repaired into
  plausible targets and never able to crash the pipeline.
- `assess_posture()` — threat-posture policy: civilian classes
  (Aircraft/Bird/Balloon) are **non-hostile at any confidence**; a confident
  drone is **requires-verification**; low confidence is **uncertain**;
  unknown classes are **unknown**. `confirmed_hostile` and
  `auto_engage_allowed` are hard-coded `False`.
- `EngagementGate` — default-deny authorization for opening the interceptor
  *simulation*: requires a requires-verification posture, an operator ID, and
  a typed confirmation phrase; grants are written to a JSONL audit log
  containing no image data, file paths, or credentials.
- `validate_upload()` / `safe_temp_upload()` — extension + size limits
  (image 20 MiB, video 200 MiB), real container suffix, guaranteed cleanup.
- `validate_checkpoint_file()` — existence/extension/size checks + SHA-256
  fingerprint logged at load, so results can be tied to exact weights.
- `js_literal()` / `validate_sim_target()` / `clamp_finite()` — safe
  embedding of values into generated HTML/JS.

Integrations: `model_manager.py` (contract validation in
`map_detection_label` and `filter_airspace_detections`, unknown classes →
"Unknown (Unverified)", longest-match class normalisation, checkpoint
fingerprinting, introspection cache, wasteful probe load removed),
`video_processor.py` (sanitization before stabilization, invalid-dimension
guard, `max_frames` resource limit, malformed-drop metrics),
`predict_image.py` (validated CLI arguments, `--name` traversal rejected,
detections routed through the same validated filter as the app),
`interceptor_sim.py` (allowlist + JSON-literal embedding, finite clamping —
data handling only; simulation physics untouched), `app.py` (posture-driven
threat panel — civilian detections get no engagement prompt; simulation
launch goes through the gate + audit log; uploads validated and cleaned up).

## 5. Evaluation

`evaluation/run_evaluation.py` runs `evaluation/cases.json` (19 cases:
normal, complex, ambiguous, contradictory, malformed, edge, resource
exhaustion, prompt injection, authorization, groundedness) against the real
code paths. **Improved code: 19/19 pass.** The same behaviours probed against
the pre-fix code (`evaluation/baseline_probe.py`) show all four legacy
defects present. This is *contract-level* evaluation — it measures how the
system handles model output, not detector accuracy.

Detector quality itself is covered by the pre-existing
`tests/test_model_evaluation.py` over `tests/artifacts/` and remains, per
the repo's own summary, at roughly precision 0.80 / recall 0.87 on 30 images
— a small sample that should be expanded before any accuracy claim is made.

Baseline test suite (this machine, CPU, Python 3.12, torch 2.5.1+cpu):
38 passed / 1 deselected (physical webcam) before the change; the full
suite including the new `tests/test_ai_safety.py` (41 tests) is reported in
the pull request / engineering report for this branch.

## 6. Model quality (Phase 2)

Measured properly (COCO-style AP, `evaluation/model_benchmark.py`), the
production checkpoint scores **mAP50 0.968 / mAP50-95 0.689** on the
test artifacts — the old "estimated mAP50 0.699" was a single-IoU proxy
that in fact approximates mAP50-95. Phase 2 found and fixed a canonical
duplicate-detection defect, adopted NMS IoU 0.45 + dedup (operating
point F1 0.835 → 0.904 at unchanged recall), and fixed the image tab's
imgsz 1280 (the worst measured setting) back to 640. Full evidence:
`docs/CV_QUALITY.md`.

## 7. Remaining limitations

- `.pt` pickle deserialization risk can only be mitigated, not removed,
  without migrating checkpoints to a safe format (e.g. ONNX/SafeTensors)
  and re-validating accuracy — recommended as the next step.
- The detector's training data and accuracy claims need a larger, documented
  evaluation set; current evidence is 30 images.
- `app.py` remains a 2,300-line monolith; the posture panel is fixed, but
  other tabs still present simulation output with marketing language
  ("100% Pk" salvo labels) that should be relabelled as simulation results.
- `.streamlit/config.toml` ships with `enableCORS = false` and
  `enableXsrfProtection = false`, which weakens browser-side protections if
  the app is network-served; re-enable them if the deployment does not
  require cross-origin embedding.
- No authentication exists in the Streamlit app; anyone who can reach the
  port can use it. Deploy only on trusted networks or behind an
  authenticating proxy.
- The engagement gate protects the *simulation launch in the image tab*;
  the standalone simulator tab itself remains directly accessible UI and is
  a visualisation only — it must never be wired to real effectors.
