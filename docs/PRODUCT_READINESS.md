# AeroWatch AI — Product Readiness Assessment (Phase 3)

Separate assessments per area; deliberately **no single readiness
score**. "Ready" below means *ready for the stated use*, nothing more.

## 1. What the product actually is (capability boundaries)

- A **YOLO object-detection application** (Streamlit) for four civilian
  aerial classes (Drone, Aircraft, Bird, Balloon), with image, video,
  and camera inputs, plus an operator-verification safety posture
  (Phase 1 `ai_safety.py` contract).
- An **HTML5 canvas simulation game** (the "interceptor") with
  scripted physics in pixel units. It is not connected to any sensor,
  weapon, or fire-control system; nothing in this project controls
  real hardware, and no real-world engagement capability exists or was
  built.
- Model predictions are **unverified classifications**, presented with
  confidence and posture; a classification is never proof of identity
  or intent, and civilian classes are never presented as hostile.

## 2. Assessments

### AI safety & reliability — **Strong for a local research tool**
Phase 1 controls intact and regression-tested (82+ tests): detection
contract validation, civilian non-hostile posture, default-deny
engagement gate with audit log, upload limits/cleanup, checkpoint
fingerprinting, injection-safe simulation embedding. Residual: `.pt`
checkpoints are pickle (code execution on load) — mitigated by
fingerprint logging, eliminated only by the ONNX migration
(recommended Phase 2, not yet integrated).

### Dataset trustworthiness — **Low**
Training data absent and unrecoverable from the repo; eval set
exploratory and potentially contaminated (same HF sources as
training; 7 eval images from a source *train* split); contradictory
training claims; two checkpoints with no provenance at all. Detail:
`docs/DATASET_PROVENANCE.md`.

### Computer-vision evidence quality — **Moderate, honestly bounded**
Real COCO-style metrics on the eval set (mAP50 0.968 / mAP50-95
0.689; operating point F1 0.904 after Phase 2 fixes) with a
reproducible harness — but the set is n=38 GT, imbalanced, possibly
contaminated, mostly clear-daylight. Figures are an upper bound, not
a certification. No GPU (RTX 3050) measurement exists from the
engineering phases; the repo's 2026-10-12 artifacts are the only GPU
evidence. Detail: `docs/CV_QUALITY.md`.

### Licensing clarity — **Now documented; legally unresolved**
The situation is fully mapped in `docs/LICENSE_AUDIT.md`: AGPL-3.0
core dependency inside an MIT-labelled repo, undeclared dataset
licenses, undocumented checkpoints. Four owner decisions are listed
there (§6). Until they are taken, **distribution beyond personal use
is not advisable**.

### Application security — **Improved; localhost-grade**
Fixed in Phase 3: `.streamlit/config.toml` now uses secure defaults
(`enableCORS = true`, `enableXsrfProtection = true`; compatibility
verified by booting and serving the app with them enabled).
pip-audit of the audit environment (94 packages): **no known
vulnerabilities in the runtime dependency set**; only `setuptools`
(build tooling) had advisories. Standing issues from Phase 1 remain:
**no authentication**, shared global model state across sessions,
unpinned dependencies, pickle checkpoint loading. Therefore:

### Distribution readiness — **Not ready**
Blocked by, in order: (1) licensing decisions (above); (2) no
authentication / session isolation; (3) dataset rights unresolved;
(4) unpinned dependencies; (5) accuracy claims constrained to an
exploratory eval set. The supported deployment today is **localhost,
single user**. Internal-network use requires an authenticating
reverse proxy or VPN (see SECURITY.md deployment policy). Public
hosting is **not approved** in the current state.

## 3. Claims discipline (Phase 3 changes)

Removed or qualified every user-facing guaranteed-outcome claim found:
hardcoded "100% Pk" strings in `app.py`, `interceptor_sim.py`, and
`video_interceptor_testbench.py` (results cards, badges, option
labels, in-sim status text, video overlay), "CERTIFIED" language for
the simulation benchmark, "guaranteed neutralization" doctrine text,
and the "Real Warplanes" tab name. A simulation disclaimer banner now
precedes the interceptor, and the README's verification note names
the unsupported figures explicitly and links the audits. The
simulation's behaviour, physics, and the Phase 1 gate were not
changed.

## 4. Outstanding readiness issues (ranked)

1. Owner licensing decision (LICENSE_AUDIT §6) — gates all
   distribution.
2. Dataset recovery + independent eval set (DATASET_PROVENANCE §4–5)
   — gates all accuracy claims.
3. Authentication strategy before any non-localhost exposure
   (established options: authenticating reverse proxy / OIDC /
   VPN — no custom auth was invented).
4. ONNX checkpoint migration — removes pickle code-execution surface.
5. Dependency pinning + pip-audit in CI — makes the license/vuln
   inventories reproducible instead of environment snapshots.
6. RTX 3050 benchmark re-run on target hardware.
