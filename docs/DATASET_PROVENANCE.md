# AeroWatch AI — Dataset Provenance & Contamination Audit (Phase 3)

Evidence standard: a fact is **Verified** only when checked against a
primary artifact in this repository or the upstream source itself
(accessed 2026-11-02). Everything else is labelled Partially verified
or Unknown. Nothing in this document was invented to fill a gap.

Reproduce the machine-checked parts with:

```
python evaluation/dataset_audit.py        # hashes, pHash, annotations, manifest
python evaluation/model_benchmark.py weights/aerowatch_v7_balanced.pt
```

## 1. Dataset inventory

| Dataset / source | Source location | Version / revision | License (see §3) | Splits upstream | Classes | How this project uses it | Verification |
|---|---|---|---|---|---|---|---|
| Rigoleto/drone-detection | huggingface.co/datasets/Rigoleto/drone-detection | HF rev `b8ad8fc3…` (last modified 2024-02-17) | **Not declared on HF**; project manifest claims CC-BY 4.0 — *unverified upstream* | `data/train`, `data/test` folders exist upstream | drone | Training source (all builders) **and** 7 of 30 evaluation images | Partially verified (HF API checked 2026-11-02; no license field present) |
| matisdsp/drone-bird-detection-dataset | huggingface.co/datasets/matisdsp/drone-bird-detection-dataset | HF rev `5f6c2789…` (last modified 2025-12-16) | **Not declared on HF**; manifest claims CC-BY 4.0 — *unverified upstream*. Filenames are Roboflow exports (`.rf.`), so original terms may live on Roboflow, not HF | `valid/` folder visible upstream (Roboflow-style export) | drone, bird (+ backgrounds) | Training source **and** 8 of 30 evaluation images | Partially verified |
| a2015003713/military-aircraft-detection-dataset | huggingface.co/datasets/a2015003713/military-aircraft-detection-dataset | HF rev `0cb9ddd0…` | **Not declared on HF**; manifest describes the eval images as "Open Images / a2015003713, CC-BY 2.0" — *unverified upstream* | Not stated | military aircraft categories (airliners extracted via category IDs 103–113 in `build_real_dataset.py`) | Training source (aircraft/jets) **and** 8 of 30 evaluation images | Partially verified |
| "Authentic Drone Flight Capture" | No URL recorded anywhere in the repo | Unknown | Manifest claims CC-BY 4.0 — *unverifiable* | Unknown | drone | 5 evaluation images (the `vgt_quadcopter_*` series) | **Unknown** |
| "Civilian Balloon Dataset" | No URL recorded anywhere in the repo | Unknown | Manifest claims CC0 — *unverifiable* | Unknown | balloon | 2 evaluation images | **Unknown** |
| Blender synthetic_v4 | Generated locally by `blender_scripts/generate_synthetic_data.py` (`.blend` scenes are in the repo) | n/a | Self-generated; underlying 3D assets' provenance not documented | n/a | balloon / "kamikaze" deltas in early builders | Training supplement for v1-era datasets; generated data absent from repo | Partially verified (generator exists; outputs absent) |
| Hard negatives (boat decks, skies, local photos) | Local files; one builder references a hardcoded personal Windows path (`C:\Users\ENG MOHAMED SABRY\...`) | Unknown | Unknown | n/a | none (negatives) | Training negatives; 5 negative evaluation images (boat deck ×4, clear sky ×1) | **Unknown** |
| Assembled training sets `data/real_aerial_v1…v4_balanced`, `data/synthetic_v1…v3` | Built by `build_*.py` from the sources above | **Absent from the repository** (gitignored `data/`) | Mixed / unresolved (inherits every row above) | Builders define train/val/test; exact membership unrecoverable | 5 raw classes: drone, aircraft, bird, balloon, military_jet | Trained every custom checkpoint | **Unverifiable** — data does not exist in the repo |

Training configuration that *does* survive: `train_v6_expert.py` /
`train_v7_balanced.py` (30 epochs, batch 8, imgsz 640, mosaic 0.7,
cosine LR for v6; v7 starts from a clean `yolo26n.pt`, v6 from
v5_pro). README's claim that *all* models trained on `real_aerial_v3`
(736 images / 1,529 instances) contradicts the v7 script's own
docstring (v4_balanced; per-class counts summing to 606 instances, no
balloon count). No `runs/train/` artifacts exist for any checkpoint.
**v5_pro and real_v1 have no training script at all.**

## 2. Contamination findings (measured, 2026-11-02)

`evaluation/dataset_audit.py` over the 35 evaluation images:

- **Exact duplicates: 0.** No SHA-256 collisions within the eval set.
- **Annotations: clean.** 38 boxes, 0 malformed lines, 0 out-of-range
  class IDs, 0 duplicate lines, 0 identical annotation sets shared
  across images.
- **Manifest integrity: verified.** All 30 manifest entries match the
  files on disk (SHA-256 and ground-truth counts) — the manifest is
  trustworthy as a record of *what the eval set contains*.
- **Near-duplicates (pHash, threshold ≤ 6/64 bits): 20 pairs flagged
  for manual review.** Interpretation matters: aerial sky imagery is
  pHash-hostile — an airliner photo and the *clear-sky negative* hash
  to distance 0, because both are mostly uniform sky. Most flagged
  pairs are this artifact, **not** evidence of duplication. The
  genuinely interesting cluster is the `vgt_quadcopter_005/030/040/
  080/115` series: five mutually near-identical drone frames,
  consistent with burst/sequential captures of one flight — i.e. the
  eval set's drone diversity is lower than the image count suggests.
  Nothing was deleted; pHash output is a review queue, not a verdict.
- **Train-split contamination — hard evidence:** 7 of the 30 eval
  images are literally named `rigo_rigoleto_train*.jpg` — they come
  from the Rigoleto dataset's **train** folder, and the builders
  train on Rigoleto. At source level, the training pipeline and the
  evaluation set demonstrably draw from the same pools for all three
  HF sources.
- **Image-level train-vs-eval comparison is impossible** because the
  training images are absent from the repo. This audit therefore
  concludes: **contamination cannot be excluded, and partial overlap
  is likely**; it does not and cannot conclude the set is clean.

**Status of the 30-image set: EXPLORATORY, POTENTIALLY CONTAMINATED.**
It is retained for historical comparison and regression testing only
(see `tests/artifacts/EVAL_SET_STATUS.md`). Scores on it are an upper
bound and must not be advertised as certified accuracy.

## 3. Dataset licensing summary

Full analysis in `docs/LICENSE_AUDIT.md` §4. Headline: **none of the
three HF datasets declares a license on Hugging Face** (no license in
cardData, no license tag — verified via the HF API on 2026-11-02).
The eval manifest's CC-BY/CC0 claims may reflect original source
pages (e.g. Roboflow for matisdsp), but they are **not corroborated
at the location the builders actually download from**. Public
accessibility is not a license.

## 4. Recommended dataset recovery process

1. **Recover before retraining.** Re-run the builders against pinned
   HF revisions (record the `sha` above), or recover the original
   `data/` trees from the training machine. Either way, commit a
   dataset *manifest* (per-image SHA-256, source, source split,
   license evidence) even if the images themselves stay out of git.
2. **Re-derive splits by source image ID**, not by file order, and
   run `dataset_audit.py` across train+eval together: exact-hash and
   pHash overlap must be **zero** before any split is called clean.
3. **Resolve or replace unlicensed sources.** For each HF dataset,
   obtain the license from the original publisher (not the HF mirror)
   or substitute a dataset with a declared license. Record the
   evidence URL and access date per source.
4. **Quarantine the current eval set** to regression duty (done:
   labelled exploratory/contaminated).

## 5. Independent evaluation set — methodology (design, not yet built)

A set may only be called *independent* when all of the following hold:

- ** disjoint sources or verified disjoint splits**: images come from
  sources/splits never fed to any builder, verified by SHA-256 +
  pHash audit against the full training manifest (both must report
  zero overlap), with the audit JSON committed alongside the set.
- **Provenance record per image:** source URL, source dataset
  revision/commit, original image ID, license + evidence URL,
  collection conditions (lighting, weather, range/size stratum,
  motion), annotator and annotation version.
- **Class balance by design:** minimum 100 instances per class
  (Drone/Aircraft/Bird/Balloon) so per-class metrics carry meaning;
  the current set's Balloon n=2 is the cautionary example.
- **Condition coverage:** quotas for small/distant objects, backlit,
  overcast, cluttered background, and motion blur — the strata the
  current set barely samples (Phase 2: 3 small, 2 distant, 1 blur).
- **Frozen and hashed:** once built, the set is frozen; any change
  increments its version and invalidates prior comparisons.
- **Rights recorded before inclusion:** no image enters the set
  without a license that permits the intended use (evaluation +
  publication of results), documented in its provenance record.

Until a set built this way exists, all accuracy communication about
this project must carry the exploratory-set caveat.
