# Licensing — Decision-Ready Summary (Phase 10)

**Status: UNRESOLVED. This document prepares the owner's decision;
it is not legal advice and grants no rights.** The LICENSE file is
unchanged (MIT label) and nothing in this phase modifies it. Full
evidence: `docs/LICENSE_AUDIT.md` and `docs/DATASET_PROVENANCE.md`.

## The facts (verified in Phases 3–5)

1. The repo is labelled MIT, but the detection stack is Ultralytics
   (`ultralytics 8.4.198`), whose own package metadata declares
   **AGPL-3.0**. YOLO11/YOLO26 carry the same terms.
2. The shipped checkpoint (v7) was custom-trained from AGPL-lineage
   base weights on datasets whose licences are **unverified** (the
   three Hugging Face sources declare no licence on their pages).
3. v5_pro and real_v1 checkpoints have **no provenance at all** and
   are marked do-not-distribute.
4. A fully license-verified alternative exists in-repo: the
   `data/verified_v1` dataset (Wikimedia Commons CC0/CC BY/CC BY-SA
   only, per-file records), but no production checkpoint has been
   trained solely on it, and its models measured weaker than v7.

## The routes

| Route | What it means | Distribution consequence |
|---|---|---|
| **A. Stay personal/research** | Keep using as-is, no distribution | No exposure; nothing can be shared or sold |
| **B. AGPL compliance** | Release the whole work (incl. weights handling per counsel) under AGPL-3.0 terms | Shareable, but copyleft obligations apply to the combined work; commercial dual-use is constrained |
| **C. Ultralytics Enterprise licence** | Purchase their commercial licence | Removes AGPL constraint for covered use; dataset provenance of v7 still unresolved |
| **D. Clean-room rebuild** | Train the release model only on `verified_v1`-style verified data with an Apache-2.0-friendly stack (e.g. YOLOX/PP-YOLOE track from the Vision planning) | Cleanest distribution story; costs a training cycle and currently measures below v7 |

## What remains legally uncertain

- Whether AGPL obligations extend to *trained model weights*
  (Ultralytics' published position says their licence covers models
  and SaaS/internal use; this has not been tested here and the
  project takes no position).
- The actual licences of the three HF training datasets (their
  pages declare none; dataset-card claims in repo manifests are
  unverified).
- Whether v7's training data included material that restricts the
  checkpoint itself.

## Recommendation on record (unchanged from Phase 3)

Remain on **Route A** until a distribution need is real; if
distribution becomes real, **Route D** is the only route whose
inputs are fully documented today, with Route C as the faster
commercial alternative pending a dataset-provenance answer.
**Owner decision required** — engineering will not pick a route,
buy a licence, or relabel the project.
