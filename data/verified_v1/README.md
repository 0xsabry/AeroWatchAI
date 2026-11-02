# verified_v1 — license-verified dataset (Phase 5)

Built because the original training data is absent and the Phase 2
evaluation artifacts are formally *exploratory / potentially
contaminated* (docs/DATASET_PROVENANCE.md). This dataset is the
project's first independently sourced, license-verified corpus.

## Source & licenses

- **Wikimedia Commons only**, fetched with
  `evaluation/fetch_commons_pool.py`, which accepts a file **only**
  when its Commons metadata carries CC0, Public domain, CC BY, or
  CC BY-SA (any version). NC/ND/unknown were rejected at fetch time.
- Every image's provenance (Commons title, page URL, author, exact
  license, license URL, SHA-256) is recorded in `MANIFEST.json`.
  CC BY / CC BY-SA images require attribution — the manifest *is*
  the attribution record; keep it with any redistribution.
- Pool: 239 files verified by content hash. Curation (visual
  grounding annotation + human verification of every flagged image)
  rejected off-domain and unboxable items (portraits matched by
  search noise, a patent drawing, an envelope interior, a contrail
  speck) and reclassified view-from-balloon / people-looking-up
  photos as hard negatives.

## Splits (frozen)

| split | images | boxes (drone/aircraft/bird/balloon) | negatives |
|---|---|---|---|
| test  | 46 | 14 / 12 / 22 / 16 | 15 |
| train | 47 (+14 background) | 12 / 13 / 13 / 89 | 14 |
| val   | 12 | 3 / 3 / 4 / 7 | 0 |

- The **test split is frozen**: never used for training, augmentation,
  threshold selection, or tuning. `build_verified_dataset.py` refuses
  to overwrite it without an explicit `--force`.
- Audits (Phase 5): **zero** exact duplicates across splits (14
  duplicate pairs found in the raw pool were removed by keeping the
  lowest index), **zero** overlap with the old tests/artifacts set,
  2 pHash near-dup flags across splits — both manually cleared as
  sky/landscape average-hash artifacts.
- Balloon train boxes are dense festival scenes (many small
  instances per image); per-class *image* counts are balanced.

## Known limitations

- 134 images total is small; per-class test GT is 12–22 boxes, so
  metrics carry wide uncertainty. Treat differences < ~0.05 mAP as
  noise.
- Commons photography skews to close-ups, ground shots, and festival
  scenes rather than distant surveillance views. This set measures
  **generalization to real-world photos**, which is precisely what
  the contaminated set could not.
- Annotations were produced by AI visual grounding with human
  verification of flagged cases, not by a professional annotation
  team; box tightness varies.
