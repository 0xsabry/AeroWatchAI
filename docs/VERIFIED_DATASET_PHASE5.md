# AeroWatch AI — Verified Dataset & Model Improvement (Phase 5)

## 1. What was recovered, and what replaced it

The original training data could not be recovered — it was never in
the repository (gitignored `data/`), and the three Hugging Face
sources the builders used declare no license upstream (Phase 3), so
re-downloading them would not have produced a *verified* dataset
anyway. Instead, a new corpus was built from **Wikimedia Commons**
with license verification at fetch time (`evaluation/fetch_commons_pool.py`:
only CC0 / Public domain / CC BY / CC BY-SA accepted; NC/ND/unknown
rejected). Full detail: `data/verified_v1/README.md`.

Final set (`data/verified_v1`, MANIFEST.json carries per-image
Commons title, author, license, source URL and SHA-256):

- test (FROZEN): 46 images, 64 boxes (Drone 14, Aircraft 12, Bird 22,
  Balloon 16) + 15 hard negatives
- train: 47 images, 127 boxes (12/13/13/89) + 14 background negatives
- val: 12 images, 17 boxes

Audits: 14 exact-duplicate pairs in the raw pool removed; **zero**
cross-split duplicates; **zero** overlap with the old
tests/artifacts set; 2 pHash cross-split flags manually cleared
(landscape average-hash artifacts). Curation rejected portraits,
a patent drawing, an envelope interior and an unboxable contrail
speck; view-from-balloon and people-looking-up photos became hard
negatives. Every flagged annotation was visually verified.

## 2. The honest v7 baseline (independent test set, adopted config)

| | old artifacts set (contaminated) | **verified_v1 test (independent)** |
|---|---|---|
| mAP50 | 0.974 | **0.497** |
| mAP50-95 | 0.651 | **0.363** |
| P / R / F1 @0.25 | 0.943 / 0.868 / 0.904 | 0.784 / 0.453 / 0.574 |
| ECE | 0.135 | 0.142 |

Per-class AP50 on independent data: Aircraft 0.785, Drone 0.537,
Bird 0.515, **Balloon 0.151 (recall 0.125 — v7 is effectively blind
to real-world hot air balloons outside its training distribution)**.
ONNX baseline: mAP50 0.486 — same picture, known FP32 drift only.

The old scores were an upper bound inflated by train/eval source
overlap, exactly as Phase 3 warned. This table is the project's
first trustworthy baseline.

## 3. Training experiments (CPU; RTX 3050 config in the script)

Script: `evaluation/train_verified_candidate.py` (seed 42,
deterministic, background negatives added as empty-label images,
early stopping patience 12; smoke test passed first). Two runs:

- **Candidate v1** — fine-tune from the clean COCO base `yolo26n.pt`,
  lr 0.01, best @ epoch 6 (stopped 18).
- **Candidate v2** — gentle fine-tune from v7 itself, lr 0.002,
  best @ epoch 12 (stopped 24). *Provenance disclosure: v2 inherits
  v7's unverifiable training history; it is an adaptation of the
  shipped checkpoint, not a clean-room model.*

### Results on the frozen test set (identical benchmark/config)

| model | mAP50 | mAP50-95 | P | R | F1 | FP | ECE | Drone | Aircraft | Bird | Balloon |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v7 (default) | 0.497 | 0.363 | 0.784 | 0.453 | 0.574 | 8 | 0.142 | 0.537 | 0.785 | 0.515 | 0.151 |
| cand. v1 | 0.551 | 0.368 | 0.244 | 0.641 | 0.353 | 127 | 0.285 | 0.361 | 0.816 | 0.642 | 0.383 |
| cand. v2 | **0.584** | 0.367 | 0.507 | 0.547 | 0.526 | 34 | **0.101** | 0.354 | 0.710 | 0.449 | **0.824** |

(AP50 columns are per-class.) Reading:

- v1 learns the new classes but is unusable at the operating point:
  127 FPs, calibration collapses (its 0.2–0.5 confidence bins are
  ≤10% precise). Small-data overfit + dense festival scenes.
- v2 fixes Balloon almost completely (0.151 → 0.824), improves
  overall mAP50 (+0.087), recall (0.453 → 0.547) and calibration
  (ECE 0.101) — **but regresses Drone (0.537 → 0.354) and Bird AP**,
  and still fires 34 FPs vs v7's 8. The Commons drone distribution
  (grounded/close-up military quads) fights v7's in-flight drone
  features; 12 training images cannot reconcile that.
- Resolution experiment: imgsz 1280 *destroys* both models on this
  set (v7 0.194, v2 0.235) — both were trained at 640. 640 stays.

## 4. Decision

**v7 remains the default checkpoint.** Under the Phase 5 gate
("meaningful improvement on the same independent set" before any
replacement), v2 is a trade, not an improvement: it wins Balloon and
aggregate mAP but loses the mission-critical Drone class. Candidate
weights are reproducible via the training script + committed run
configs (weights themselves are not committed, per the
no-weight-distribution constraint); v2's ONNX export was verified
for parity (mAP50 0.576 vs 0.584 torch; per-class within ±0.02).

**What would actually move Drone**: verified *in-flight* drone
imagery at surveillance-like distances — scarce on Commons. The
highest-value next step is a larger verified harvest (more queries,
Open Images airplane/bird subsets under CC-BY) plus 3–5× more
annotated in-flight drone frames, then re-run this exact protocol.

## 5. Constraints honoured

No safety code touched; both candidates were evaluated through the
unchanged ai_safety/filter path. v7 checkpoint byte-identical
(SHA-256 `992080afb8f38674…`). Test split never used for training or
tuning. No GPU was available: all training/benchmarks are CPU
measurements; RTX 3050 figures remain measurable with
`python evaluation/hardware_benchmark.py` and, for training, the
same train script on the GPU machine (batch 8 @ 640 fits 4 GB).
