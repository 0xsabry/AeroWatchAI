# AeroWatchAI — Documentation Index

Start with the **[Final Release Guide](FINAL_RELEASE_GUIDE.md)** — it
is the single entry point. This index organises everything else.
Documents marked *(authoritative)* are the current source of truth
for their topic; phase reports are historical evidence and are kept
for reproducibility.

## Getting started

| Document | What it covers |
|---|---|
| [Final Release Guide](FINAL_RELEASE_GUIDE.md) *(authoritative)* | The whole product: install → workflows → history → backup → licensing status |
| [Owner Finalization Checklist](OWNER_FINALIZATION_CHECKLIST.md) | Every remaining owner action: hardware acceptance, GitHub settings, licensing decision, sign-off |
| [Windows Guide](WINDOWS_GUIDE.md) | Windows install/launch, CPU & GPU paths, offline mode, troubleshooting |
| [Release Candidate Notes (Phase 9)](RELEASE_CANDIDATE_PHASE9.md) | RC checklist, storage behaviour, known limitations |
| [Release Manifest](RELEASE_MANIFEST.json) | Machine-readable freeze record: hashes, versions, three-state status |

## Architecture & AI engineering

| Document | What it covers |
|---|---|
| [Architecture](ARCHITECTURE.md) *(authoritative)* | Components, data flow, safety pipeline, storage design |
| [AI Engineering](AI_ENGINEERING.md) | The safety contract, posture model, engagement gate, validation rules |
| [LLM & Prompt Audit (Phase 6)](LLM_PROMPT_AUDIT_PHASE6.md) | Proof that no LLM exists in the system; the permanent boundary rules |
| [Product Integrity (Phase 7)](PRODUCT_INTEGRITY_PHASE7.md) | Truthful-presentation corrections and their regression guards |
| [Product Evolution (Phase 8)](PRODUCT_EVOLUTION_PHASE8.md) | Dashboard, history, analytics design decisions |

## Computer vision & data

| Document | What it covers |
|---|---|
| [Verified Dataset (Phase 5)](VERIFIED_DATASET_PHASE5.md) *(authoritative for metrics)* | The independent test set and the honest v7 baseline (mAP50 0.497) |
| [CV Quality (Phase 2)](CV_QUALITY.md) | Benchmark harness, canonical-dedup defect, adopted configuration |
| [Dataset Provenance](DATASET_PROVENANCE.md) | Where every dataset came from; contamination findings |
| [Runtime & ONNX (Phase 4)](RUNTIME_PHASE4.md) | Backends, parity evidence, dependency locking, hardware benchmark |

## Licensing, security, readiness

| Document | What it covers |
|---|---|
| [Licensing Decision](LICENSING_DECISION.md) *(authoritative)* | The open decision: routes, matrix, uncertainties |
| [License Audit](LICENSE_AUDIT.md) | Evidence behind the licensing position |
| [Product Readiness](PRODUCT_READINESS.md) | Deployment verdicts (localhost / proxied / public) |
| [SECURITY.md](../SECURITY.md) | Security policy and the AI safety boundaries |

## Historical phase reports

Phases 1–11 each left a report in the repository history and in the
documents above. Older standalone reports (`AUDIT_REPORT.md`,
`BENCHMARK_REPORT.md`, `VIDEO_AND_ACCURACY_REPORT.md` at the repo
root) are **superseded** — they carry banners saying so and are kept
only as historical record. Where any older document disagrees with
the *(authoritative)* documents above, the authoritative one wins.
