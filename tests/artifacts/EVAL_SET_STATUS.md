# Evaluation set status — READ BEFORE CITING ANY SCORE

The 30 annotated images + 5 negatives in this directory are an
**exploratory, potentially contaminated** evaluation set. They are
**not** an independent, provenance-clean test set:

- Their sources (Rigoleto/drone-detection, matisdsp/drone-bird,
  a2015003713 via Open Images, per `manifest.json`) **overlap the
  sources the training dataset builders draw from**
  (`build_real_dataset.py`, `build_dataset_v3.py`,
  `build_balanced_dataset.py`).
- Seven images carry filenames from a source dataset's **train** split
  (`rigo_rigoleto_train*.jpg`).
- The original training data is absent from this repository, so
  overlap can neither be confirmed nor excluded image-by-image.
- The set is small and imbalanced (38 boxes: Drone 16, Aircraft 16,
  Bird 4, Balloon 2); per-class figures for Bird/Balloon are
  indicative only.

Scores measured on this set (see `docs/CV_QUALITY.md`) are an
**upper bound** on real-world performance and must not be advertised
as certified accuracy. The set is retained for historical comparison
and regression testing. The methodology for building a genuinely
independent set is in `docs/DATASET_PROVENANCE.md`. Run
`evaluation/dataset_audit.py` to regenerate the audit evidence.
