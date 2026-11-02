# AeroWatch AI — Runtime Modernization (Phase 4)

## 1. ONNX backend (opt-in, PyTorch remains default)

- `weights/aerowatch_v7_balanced.onnx` — FP32 export of the unchanged
  v7 checkpoint (SHA-256 of the .pt before/after export identical:
  `992080afb8f38674…`; ONNX file SHA-256 `a45339daaaac6789…`).
  Regenerate: `python evaluation/export_onnx.py`.
- `onnx_backend.py` — ONNX Runtime session with Ultralytics-style
  letterbox preprocessing and numpy NMS. It implements the same
  `predict()` surface the codebase uses, so its raw candidate boxes
  flow through the **identical** `ai_safety` contract, canonical
  mapping, geometric filters and dedup in
  `ModelManager.filter_airspace_detections`. No path bypasses
  `ai_safety.py`.
- Use it via `ModelManager.load_onnx_model("weights/aerowatch_v7_balanced.onnx")`
  (`mm.backend == "onnxruntime"`). Missing onnxruntime produces a
  clear RuntimeError with install instructions, never a stack trace.
- **Default is unchanged.** Switching the default requires the parity
  gates (below) plus an explicit owner decision.

### Parity investigation (the Phase 2 "35/37" question, resolved)

- Raw-output comparison on an identical input tensor: max abs diff
  **1.9e-03**, mean 1.6e-05 — pure FP32 kernel-order drift between
  PyTorch and ONNX Runtime. There is no export bug.
- Through the full app filter path (conf 0.25, NMS 0.45, dedup):
  PyTorch produced 35 canonical detections, ONNX 38, and **all 35
  PyTorch detections were matched** by ONNX. The 3 ONNX-only
  detections are all borderline *Uncertain*-band candidates
  (conf 0.286 / 0.314 / 0.351) whose scores drift across the 0.25
  threshold — one is a low-confidence drone on a negative (boat deck)
  image. **Zero confident detections differ; zero PyTorch detections
  are lost.** Phase 2's 2 unmatched detections were the same
  phenomenon from the other direction.
- Full-set metrics (same benchmark, iou 0.45 + dedup): ONNX
  mAP50 0.986 / mAP50-95 0.672, operating point P/R/F1
  0.921/0.921/0.921 vs PyTorch 0.974 / 0.651 / 0.904 — statistically
  indistinguishable at n=38 GT; the small deltas are the borderline
  flips above, not a quality change.
- Load time: **0.7–0.8 s (ONNX) vs 3.9–4.6 s (PyTorch)** in every
  run. Per-image CPU latency in this shared sandbox is within
  run-to-run noise between backends; use
  `evaluation/hardware_benchmark.py` on target hardware for a clean
  comparison.
- Gates: `tests/test_onnx_parity.py` (backend identity + class
  vocabulary, confident-detection parity through the safety filter,
  contract compliance, graceful missing-dependency error).

### Known ONNX backend limits

- `track()` is not implemented (video tracking workflows keep the
  PyTorch backend); the wrapper says so explicitly.
- The COCO person-verifier inside the filter still uses the PyTorch
  `yolo26n.pt` — verifier behaviour is therefore identical across
  backends.
- CUDA: the wrapper selects ONNX Runtime's CUDA provider when
  available (onnxruntime-gpu) and otherwise falls back to CPU with a
  logged warning. Not testable in the Phase 4 sandbox (no GPU).

## 2. Dependency reproducibility

- `requirements.txt` — direct dependencies **pinned** to the tested
  versions (previously all `>=` minimums).
- `requirements-lock.txt` — full `pip freeze` of the tested
  environment (Python 3.12, CPU), including the PyTorch CPU index
  line. Verified by installing into a **clean venv**: install OK,
  `pip check` clean, app imports, ONNX loads, contract evaluation
  19/19, full suite green (see Phase 4 report).
- **CPU install:** `pip install -r requirements-lock.txt`
- **GPU install (e.g. RTX 3050):** install CUDA torch first, e.g.
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
  (pick the CUDA build matching your driver), then
  `pip install -r requirements-lock.txt` — pip will keep the CUDA
  torch already installed only if versions match the lock; simpler
  and safer on GPU machines: install CUDA torch, then
  `pip install -r requirements.txt` (direct deps pinned; torch lines
  refer to CPU builds, so install requirements with `--no-deps` for
  torch already present, or edit locally). When in doubt, the tested
  path is the CPU lock; GPU version selection is hardware-specific
  and intentionally not locked here.
- Dev/audit tools (flake8, pip-audit) are CI tools, pinned in the
  workflow, not in the runtime lock. `onnx` (export-time only) is
  likewise excluded from the runtime lock; `onnxruntime` is a runtime
  dependency of the ONNX backend and is included.

## 3. CI (`.github/workflows/ci.yml`)

Now runs, in order: locked install → `pip check` consistency →
**pip-audit** vulnerability scan (fails the build on any finding; no
ignore list) → **license inventory validation** (coverage of the
runtime set + AGPL recorded) → flake8 (file list extended with
`ai_safety.py`, `onnx_backend.py`, `predict_image.py`,
`evaluate_videos.py`) → import smoke → full pytest suite (including
ONNX parity gates on the committed export) → AI contract evaluation.
`scripts/ci_checks.sh` mirrors the same gates locally.

## 4. Hardware benchmark (RTX 3050)

`python evaluation/hardware_benchmark.py` — detects the actual GPU,
runs the real app pipeline (predict + safety filter) over the test
artifacts for torch FP32/FP16 and ONNX (CUDA provider when present),
and writes `evaluation/results/hardware_benchmark.json` + `.md`
(load time, warm-up, mean/p50/p95 latency, FPS, GPU peak memory, RSS,
detections/image). On a machine without CUDA it reports that honestly
and skips FP16/GPU rows instead of fabricating numbers.
