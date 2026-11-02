# AeroWatchAI — Owner Finalization Checklist (Phase 12)

Everything engineering could verify without your hardware, your
GitHub settings access, or your legal decision is done. What remains
is below — four categories, each action with exact steps, the exact
command, the expected output, how to tell it succeeded, and whether
it needs a human decision.

**Current frozen status**

| | |
|---|---|
| Technical verification | ✅ **TECHNICALLY VERIFIED** (169 tests, eval 19/19, browser QA, clean install) |
| Owner validation | ⏳ **OWNER VALIDATION PENDING** (Category A + B below) |
| Distribution | ⛔ **DISTRIBUTION NOT CLEARED** (Category C decision required) |

Freeze record: `docs/RELEASE_MANIFEST.json` · Guide:
`docs/FINAL_RELEASE_GUIDE.md`

---

## A. Windows and hardware (your RTX 3050 machine)

### A1. First PowerShell installation and launch
- **Steps:** clone/pull the repo → open PowerShell in the folder → run setup, then launch.
- **Commands:**
  ```powershell
  powershell -ExecutionPolicy Bypass -File setup_windows.ps1
  powershell -ExecutionPolicy Bypass -File launch_windows.ps1
  ```
- **Expected:** setup ends with preflight all PASS/WARN (no FAIL)
  and "Setup complete"; the app opens at http://localhost:8501.
- **Success:** Dashboard loads; sidebar checkpoint shows
  `aerowatch_v7_balanced.pt`; `python preflight.py` prints
  `Overall: OK` and the checkpoint line says the SHA-256 *matches
  the verified release hash*.
- **Human approval:** none — safe to run; it installs only the
  pinned CPU dependencies into `.venv` and never downloads weights.

### A2–A5. One-command acceptance (diagnostics, GPU benchmark, camera, offline)
Run the unified validator in full mode — it orchestrates the
existing tools; do not run the stages separately unless one fails.

- **Command:**
  ```powershell
  powershell -ExecutionPolicy Bypass -File validate_release.ps1 -Full -Camera -Offline
  ```
- **What each stage validates:**

  | Stage | Validates |
  |---|---|
  | preflight | Python, dependencies, checkpoint SHA-256, storage writability |
  | dependencies | `pip check` consistency against the installed environment |
  | hardware_info | CPU/RAM/GPU/driver/CUDA/ONNX providers actually present |
  | onnx_smoke | ONNX artifact loads with the correct class map |
  | database | History write → backup → clear → restore → integrity roundtrip |
  | app_startup | Real Streamlit boot; health endpoint returns 200 |
  | hardware_benchmark (`-Full`) | Measured inference on the real pipeline: PyTorch + ONNX, FP32/FP16, latency percentiles, FPS, VRAM, load time → `evaluation/results/hardware_benchmark.json/.md` |
  | camera (`-Camera`) | Physical webcam: enumeration, frame acquisition, grab latency, session lifecycle, history roundtrip; **zero frames saved** |
  | offline_mode (`-Offline`) | External socket connections are blocked while local operation continues |

- **Expected output:** a per-stage `[PASS] / [FAIL] / [NOT VERIFIED]`
  list plus `release_validation/report.json` and `report.md`.
- **Success:** no `[FAIL]` lines. `hardware_benchmark` reports
  `gpu_measured: true` (it prints the device it measured — on the
  RTX 3050 that must be the GPU, with FP16 results and peak VRAM
  recorded). `camera` reports PASS only if a real webcam completed
  the lifecycle; with no camera attached it honestly stays
  NOT VERIFIED — that is a finding, not a failure.
- **Human approval:** none for running it. **Acceptance criteria**
  are outcome-based: benchmark completes at imgsz 640 without
  out-of-memory errors; no target FPS/latency figures were promised
  in advance — the measured numbers become the record.

### A6. Judge the GPU numbers
- **Steps:** open `evaluation/results/hardware_benchmark.md` after A2–A5.
- **Success:** you accept the measured latency/FPS for your use, or
  record them as insufficient — either way they are now *measured*,
  replacing the "unverified" status for hardware.
- **Human approval:** **YES — your judgement call.**

---

## B. GitHub (settings only you should change)

### B1. Upload the social preview image
- **Steps:** GitHub → repo **Settings → General → Social preview →
  Upload an image…** → choose `assets/social-preview.png`
  (1280×640, already in the repo). GitHub exposes no API for this
  setting; it is manual by design.
- **Expected:** the preview card shows the navy AeroWatchAI banner.
- **Success:** share the repo URL anywhere (chat/social) and the
  banner card appears.
- **Human approval:** **YES — settings change on your account.**

### B2. Apply the CI workflow
- **Steps (either):** (a) GitHub web UI → `.github/workflows/` →
  Add file → upload `docs/ci_workflow_phase4.yml` as `ci.yml`; or
  (b) grant the connected GitHub token the `workflow` scope and ask
  for it to be pushed. (Pushes from the engineering environment are
  rejected without that scope — this is a platform rule, verified
  in Phase 9, not an error to retry.)
- **Expected:** Actions tab shows the workflow; the first run
  executes the same gates as `scripts/ci_checks.sh`.
- **Success:** a green run on any branch.
- **Human approval:** **YES.**

### B3. Verify description and topics (already set — just confirm)
- **Current state (set during the presentation phase):**
  description = *"Local-first aerial object detection and
  simulation — YOLO computer vision with a deterministic AI-safety
  layer, persistent detection history, analytics, and honestly
  measured benchmarks. No cloud, no LLM."*; topics = computer-vision,
  object-detection, yolo, pytorch, onnx, python, streamlit,
  drone-detection, video-processing, machine-learning, deep-learning.
- **Success:** the About panel matches; edit only if you want
  different wording.
- **Human approval:** only if you change it.

### B4. After any future authorized merge to the default branch
- **Steps:** open the repo home and confirm the premium README
  renders (banner, badges, screenshots, Mermaid diagram — all
  verified rendering correctly on the feature branch).
- **Human approval:** **YES — merging is your decision;** no merge
  has been made or requested by engineering.

---

## C. Licensing (your decision — engineering will not choose)

Full analysis: `docs/LICENSING_DECISION.md` (final matrix included).
Summary of the four routes:

| Route | What you do | Consequence |
|---|---|---|
| **A. Personal/research** | Nothing | Use stays private; no distribution of app or weights |
| **B. AGPL compliance** | Release the combined work under AGPL-3.0 terms (get counsel) | Copyleft obligations; weight status still legally uncertain |
| **C. Ultralytics Enterprise** | Purchase their commercial licence | Removes the AGPL constraint for covered use; v7 training-data provenance still unverified |
| **D. Clean-room rebuild** | Commission training only on license-verified data (the `verified_v1` pattern) | Cleanest distribution story; current verified-data models measure below v7 |

- **Success:** a route is chosen and recorded; only then may the
  distribution status change.
- **Human approval:** **YES — required. Until then the correct
  status remains DISTRIBUTION NOT CLEARED, and no MIT badge or
  commercial claim may be added.**

---

## D. Release acceptance (sign-off)

| Gate | State | What closes it |
|---|---|---|
| Technical verification | ✅ TECHNICALLY VERIFIED | — (frozen: `docs/RELEASE_MANIFEST.json`) |
| Owner validation | ⏳ PENDING | Category A completed on your machine (A1–A6) |
| Distribution clearance | ⛔ NOT CLEARED | Category C decision (+ B2 if you want CI-gated releases) |

- **Final acceptance command (one line, on your machine):**
  ```powershell
  powershell -ExecutionPolicy Bypass -File validate_release.ps1 -Full -Camera -Offline
  ```
- **Success:** the validation report shows no FAIL; you record the
  GPU benchmark as the hardware record; the licensing route is
  written down. At that point the release candidate is fully
  owner-validated — and distribution still follows only from the
  Category C route you chose.
- **Human approval:** **YES — final sign-off is yours.**
