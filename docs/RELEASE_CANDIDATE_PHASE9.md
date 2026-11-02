# AeroWatch AI — Release Candidate Notes (Phase 9, 2026-11-02)

> **Frozen status (Phase 11, authoritative):** TECHNICALLY VERIFIED ·
> OWNER VALIDATION PENDING · DISTRIBUTION NOT CLEARED — see
> `docs/RELEASE_MANIFEST.json` and `docs/FINAL_RELEASE_GUIDE.md`.
> These Phase 9 notes are the historical RC record.

Status: **release candidate for local/internal evaluation. NOT
cleared for commercial distribution** — the licensing decisions in
`docs/LICENSE_AUDIT.md` §6 remain open, and the LICENSE file is
unchanged.

## Install (verified procedure)

```bash
python -m venv .venv
# CPU (any machine):
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA machine): install a CUDA torch build first, then:
pip install -r requirements-lock.txt
streamlit run app.py        # http://localhost:8501
```

`requirements.txt` pins the direct dependencies;
`requirements-lock.txt` is the fully-tested freeze. Python 3.12 was
used for all verification. The optional ONNX backend needs
`onnxruntime` (declared in both files) and the exported
`weights/aerowatch_v7_balanced.onnx`; PyTorch is the default.

## Storage behaviour

- Detection history: local SQLite at `runs/history/` (gitignored),
  override with `AEROWATCH_HISTORY_DIR`. Metadata only — no images,
  videos, or camera frames are ever stored. Retention default 30 days
  (configurable in Detection History), session delete and clear-all
  are confirmation-gated in the UI.
- Model runs/detection outputs follow the existing `runs/` layout.

## Supported inputs

- Images: JPG/PNG/WEBP uploads (validated, size-capped) or bundled
  samples.
- Video: MP4/AVI/MOV uploads ≤ 200 MB (validated before processing),
  bundled presets (two are AI-generated and labelled synthetic).
- Camera: local webcam via OpenCV backends, where hardware exists.

## Model performance (documented, reproducible)

Independent verified test (Phase 5, `evaluation/results/
bench_v7_independent_test.json`): **mAP50 0.497, mAP50-95 0.363,
P/R/F1 @0.25 = 0.784/0.453/0.574, ECE ≈ 0.14** on 46 images / 64
boxes. Per-class AP50: Aircraft 0.785, Drone 0.537, Bird 0.515,
Balloon 0.151. Older "contaminated-set" numbers are an upper bound
only. GPU (RTX 3050) latency has still **not** been measured by the
project — run `python evaluation/hardware_benchmark.py` on the
target machine for real figures.

## Security & privacy boundaries

Localhost-first. The app has **no authentication**; for any shared
access, run it behind an authenticating proxy/VPN (see
`docs/PRODUCT_READINESS.md`). Detections are unverified predictions;
civilian classes are never hostile; engagement requires the
operator gate (ID + typed phrase + audit log) inside a clearly
labelled simulation. Exports neutralise spreadsheet formula
injection. History never leaves the machine.

## Known limitations

- No persistent object tracking (per-frame detection + box smoothing).
- Camera sessions record aggregate counts only; in this development
  environment the lifecycle was verified with simulated detections
  and code-level tests — **no physical camera was available**, so
  real-hardware streaming remains unverified.
- Small verified dataset: metric differences under ~0.05 mAP are
  noise; Balloon detection is weak.
- GitHub Actions: the upgraded workflow still cannot be pushed from
  this environment (token lacks `workflow` scope). **Owner action:**
  copy `docs/ci_workflow_phase4.yml` to `.github/workflows/ci.yml`
  (or grant the scope). The local mirror `scripts/ci_checks.sh`
  passes all gates.

## Release checklist (Phase 9 verification)

- [x] Full test suite green (see Phase 9 report for counts)
- [x] Contract evaluation 19/19; CI mirror all-pass; pip-audit clean
- [x] v7 checkpoint SHA-256 unchanged (`992080af…`)
- [x] Real-browser QA: 7 pages × 5 viewports (Firefox/Playwright),
      screenshots in `docs/qa_phase9/`
- [x] Image workflow end-to-end in browser (detect → explain →
      record → dashboard → export)
- [x] History numbers cross-validated against direct SQL
- [ ] Physical camera verification (no hardware in dev environment)
- [ ] RTX 3050 hardware benchmark (owner machine)
- [ ] Licensing decisions (owner) — blocks any distribution
- [ ] GitHub Actions workflow applied (owner, see above)
