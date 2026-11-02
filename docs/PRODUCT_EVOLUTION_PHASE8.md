# Phase 8 — Product Evolution (2026-11-02)

Branch `feat/product-evolution-phase8` (from Phase 7 `39b77c9`).

## Prioritised plan (from the UX assessment) and what shipped

The weakest product experiences were: no landing/overview page (the app
opened straight into a tool page), **no memory** — every analysis
vanished on rerun — weak per-prediction explanation, a video workflow
that defaulted to the checkpoint's *worst* measured resolution, and no
way to export anything. Shipped, in that priority order:

1. **Persistent detection history** (`history_store.py`, stdlib SQLite,
   no new dependencies). Sessions + prediction rows; metadata only —
   no media bytes, no full paths (basename labels), no operator IDs.
   Per-thread connections (Streamlit shares cached resources across
   session threads; WAL mode). Case-insensitive class filtering,
   confidence/date/search filters, explicit human verification marking
   (`unverified` by default; the store rejects any status outside
   unverified/verified/rejected), per-session caps, retention pruning
   (default 30 days, configurable in the UI), session delete and
   clear-all behind explicit confirmations. Video runs record honest
   **aggregate class counts** on the session — individual rows are
   never fabricated for frames that were not stored.
2. **Dashboard** (new landing page): session/detection/unverified
   totals, class distribution, confidence histogram, detections over
   time, recent sessions, checkpoint/backend/SHA panel — all computed
   from the real store (`reporting.dashboard_metrics`), with an honest
   empty state and a clear separation between *this browser session*
   and *long-term local history*.
3. **Detection History page**: filters (class, confidence range,
   search, session), CSV export of the filtered set, per-session
   Markdown report + JSON record + full-history JSON export (all
   carrying the unverified-prediction disclaimer), verification
   marking, deletion and retention controls.
4. **Analytics & Reports page**: history charts, per-session
   comparison, human-review tallies (labelled as operator judgements,
   not ground truth), and documented model performance read from the
   committed Phase 5 benchmark artifact — operational history is never
   used to invent precision/recall.
5. **Explainability**: every image analysis now has a "Prediction
   details" panel — per detection: class, detector score, posture,
   box, frame coverage, checkpoint + backend, and a plain-language
   confidence-band explanation (`reporting.confidence_band`) including
   *why* a prediction is marked uncertain.
6. **Video workflow fixes**: uploads are validated (type + 200 MB cap)
   *before* storing/processing, with size shown; the inference-size
   default is now **640** (the checkpoint's trained, measured-best
   resolution) instead of 1280, with honest option labels; processed
   runs are recorded in history automatically.
7. **Recording integrity**: image analyses are recorded exactly once
   per unique (source, thresholds, checkpoint) fingerprint, so
   Streamlit reruns cannot duplicate history. Recording failures never
   block detection.

## Not changed

`ai_safety.py` (byte-identical), model defaults, checkpoint (v7
SHA-256 verified), backends (PyTorch default, ONNX optional), the
engagement gate, simulation behaviour, licensing. No LLM, no new
dependencies, no external calls: history is a local file
(`runs/history/`, gitignored; `AEROWATCH_HISTORY_DIR` override).

## Verification

- New tests: `tests/test_history_store.py` (11),
  `tests/test_reporting.py` (6),
  `tests/test_product_evolution_phase8.py` (6).
- **Streamlit AppTest** (headless, real script execution): Dashboard,
  Detection History, Analytics & Reports and Optical Image Analysis
  all run with **zero exceptions**; the image page's live run detected
  the sample drone (0.74, requires_verification) and the session was
  found in the history database afterwards — the full
  detect → posture → record → query loop is proven end to end.
- Full suite + contract evaluation + CI mirror: see the Phase 8
  completion report.

## Known limitations / deferred

- Interactive browser click-through (a human-driven browser session)
  was not performed; AppTest executes the real script logic headlessly
  and the server boots (health 200), but visual layout was verified
  only by construction, not by screenshot.
- Camera-stream sessions are not recorded (continuous stream; no
  natural session boundary) — image and video runs are.
- PDF export: no PDF library exists in the pinned dependency set and
  none was added; Markdown + JSON + CSV cover the reporting scope.
- History is single-user local storage by design; there is still no
  authentication (deployment guidance unchanged: local or behind an
  authenticating proxy).
