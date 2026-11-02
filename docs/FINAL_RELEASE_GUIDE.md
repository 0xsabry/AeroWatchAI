# AeroWatch AI — Final Release Guide (Phase 11)

**Release status: TECHNICALLY VERIFIED · OWNER VALIDATION PENDING ·
DISTRIBUTION NOT CLEARED.** The software is verified end-to-end in
the development environment; GPU/camera/Windows-script acceptance
runs remain the owner's, and licensing blocks distribution
(§14). This guide is the single entry point; details live in the
linked documents.

## 1. What AeroWatchAI actually does

A local, single-user web app (Streamlit) that runs YOLO object
detection for a 4-class civilian airspace vocabulary (Drone,
Aircraft, Bird, Balloon) on images, videos, and camera frames;
applies a deterministic safety layer (civilian classes are never
hostile; a confident drone requires operator verification; nothing
is ever auto-engaged); records a local detection history; and
includes a clearly-labelled browser **simulation game** (not a
fire-control system). There is **no LLM, no cloud service, and no
object tracking** (per-frame detection + box smoothing only).
Measured performance on the independent verified test set:
**mAP50 0.497, per-class AP50 from 0.785 (Aircraft) down to 0.151
(Balloon)** — see `docs/VERIFIED_DATASET_PHASE5.md`.

## 2. Installation

Windows: `setup_windows.ps1`, then `preflight.py` must pass.
Any platform: venv + `requirements-lock.txt`. Guide:
`docs/WINDOWS_GUIDE.md`. Internet is needed only for pip.

## 3. Startup and shutdown

`launch_windows.ps1` (or `streamlit run app.py`) →
http://localhost:8501. Ctrl+C stops cleanly. Unified acceptance:
`validate_release.ps1` (quick) / `-Full -Camera -Offline`.

## 4. CPU and GPU configuration

CPU works out of the box. For an RTX 3050: the setup script prints
the exact CUDA torch command (never installs it silently); keep
inference size **640** (trained resolution, measured best). Verify
with `evaluation/hardware_diagnostics.py`, measure with
`evaluation/hardware_benchmark.py`. PyTorch is the default backend;
ONNX is opt-in (`docs/RUNTIME_PHASE4.md`).

## 5. Image and video workflows

Uploads are validated (type/size) before processing. Image results
include a per-prediction explainability panel. Video runs record
aggregate class counts — never invented per-frame rows. Two bundled
video presets are AI-generated and labelled synthetic.

## 6. Camera usage

Start the sensor on the camera page; a session tracker records
duration + per-frame prediction aggregates into history on stop
(interrupted sessions are recorded as interrupted). Validate a
physical webcam with `evaluation/camera_check.py --model`.
**Physical-camera verification is still pending** (no camera in the
development environment).

## 7. Detection history

Local SQLite (`runs/history/`, metadata only — no media, no full
paths, no operator IDs). Searchable/filterable in the app; retention
defaults to 30 days. Architecture: Phase 8 notes in
`docs/PRODUCT_EVOLUTION_PHASE8.md`.

## 8. Analytics and exports

Dashboard + Analytics pages compute from real history only;
documented model performance comes from the committed benchmark
artifact. CSV/JSON/Markdown exports carry the unverified-prediction
disclaimer; CSV cells are formula-injection neutralised.

## 9. Backup and restore

History page → *Create backup now* (SQLite online backup), or
`HistoryStore.backup_to()`. Restore is a confirmed maintenance
operation that validates the backup and preserves the current DB
first: `docs/WINDOWS_GUIDE.md` §Data.

## 10. Offline operation

No runtime network is required. `-Offline` / `AEROWATCH_OFFLINE=1`
blocks non-loopback sockets process-wide (Phase 10).

## 11. Troubleshooting

`docs/WINDOWS_GUIDE.md` has the table (Python missing, pip failure,
checkpoint hash mismatch, port busy, camera privacy, slow CPU).

## 12. Privacy and security

Localhost-first; **no authentication** — shared access only behind
an authenticating proxy/VPN (`docs/PRODUCT_READINESS.md`). Uploads
are validated and sandboxed by the safety layer (`ai_safety.py`);
engagement in the simulation requires operator ID + typed phrase
and is audit-logged.

## 13. Current AI limitations

Balloon detection is weak (AP50 0.151); recall overall is 0.453 on
the verified set; detector scores are not calibrated probabilities
(ECE ≈ 0.14); the verified dataset is small, so small metric
differences are noise; no tracking; no distance estimation.

## 14. Licensing restrictions

**Distribution is NOT cleared.** The stack is Ultralytics
(AGPL-3.0 per its own metadata) inside an MIT-labelled repo, and
the default checkpoint's training data has unverified licences.
Decision matrix: `docs/LICENSING_DECISION.md`; evidence:
`docs/LICENSE_AUDIT.md`. Do not redistribute the app or weights
until the owner picks a route.

## 15. Known unverified hardware capabilities

- RTX 3050 inference performance (FP32/FP16, VRAM) — owner runs
  `hardware_benchmark.py`; acceptance = completes without OOM at
  batch 1/imgsz 640 and reports its own numbers (no target figures
  are promised in advance).
- Physical webcam streaming — owner runs `camera_check.py --model`.
- PowerShell scripts' first execution on Windows (their logic is
  mirrored by the tested `preflight.py` / `release_validation.py`).
- GitHub Actions — workflow preserved at
  `docs/ci_workflow_phase4.yml`; applying it is an owner action
  (token lacks `workflow` scope).
