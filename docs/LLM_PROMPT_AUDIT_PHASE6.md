# AeroWatch AI — Phase 6: LLM & Prompt Engineering Audit and Design

**Date:** 2026-11-02 · **Branch:** `feat/llm-prompt-audit-phase6` (from
`feat/verified-cv-model-improvement-phase5`, commit `d6e66e8`)
**Default checkpoint:** `weights/aerowatch_v7_balanced.pt`, SHA-256
`992080afb8f38674ddbf4eb84b9f8123b89bb4f90965c52e6c5e58af328b51cb`
(verified unchanged before and after this phase).

## 0. Headline verdict

**AeroWatch AI contains no LLM, no prompt layer, no LLM provider SDK, no
API key, and no external AI API. There is nothing to "tune".** Every
operator-facing sentence the system produces today is deterministic code:
enum postures, f-string templates, static HTML, and a typed-confirmation
gate. Phase 1's `docs/AI_ENGINEERING.md` already stated this; Phase 6
re-verified it exhaustively (Section 2) and codified it as a regression
test (`tests/test_llm_boundary.py`) so an LLM dependency can never land
silently.

The honest engineering question for this phase is therefore not "how do
we improve the prompts" but: **(a)** are the *deterministic* language
surfaces that already exist accurate, grounded and injection-safe
(Sections 3–5 — two real defects found and fixed); **(b)** if an LLM is
ever added, exactly where may it help, where must it never be, and under
what architecture, governance, schema validation and evaluation would it
be safe (Sections 6–14, **design only — nothing implemented**); and
**(c)** what must the owner decide first (Section 16).

Scores (this phase's scope):

| Dimension | Verdict |
|---|---|
| LLM / prompt inventory accuracy | **Complete — zero LLM surface exists** |
| Deterministic text groundedness | **Moderate** — one fabricated-results block found & fixed (§4.1); README body claims remain ungrounded behind a disclaimer (§4.2) |
| Injection resistance of current text surfaces | **Strong for its class** — allowlists, `js_literal`, HTML escaping; residual risk concentrated in 66 `unsafe_allow_html` call sites (§5) |
| Privacy of AI inputs today | **Strong** — nothing leaves the machine at runtime (§11) |
| Readiness to add an LLM | **Not ready, by design** — architecture, schemas, governance and harness are specified here; implementation is a separate, gated phase (§15) |

---

## 1. Scope, method and evidence standard

Method (all commands run against the working tree at `d6e66e8` + this
phase's fixes; reproducible with `grep`):

1. Full-text searches over all 56 Python files plus Markdown, TOML, YAML
   and JSON for: provider/SDK names (`openai|anthropic|claude|gpt|gemini|
   mistral|cohere|langchain|transformers|ollama|groq|…`), prompt
   vocabulary (`prompt|system_prompt|temperature|max_tokens|few-shot|…`),
   network clients (`requests|urllib|httpx|fetch|websocket`), secrets
   (`api_key|token|secret|environ|getenv|dotenv`) and credential shapes
   (`sk-…|AKIA|ghp_|-----BEGIN`).
2. A complete import inventory of every Python module, compared against
   `requirements.txt` (pinned, Phase 4) and `requirements-lock.txt`.
3. Manual trace of the detection→decision→display data flow through
   `model_manager.py`, `ai_safety.py`, `app.py`, `predict_image.py`,
   `video_processor.py`, `interceptor_sim.py`.
4. Git filename history scan (all branches in the local clone) for
   secret-bearing files (`.env`, keys, credentials).
5. Re-run of the full test suite and the Phase 1 contract evaluation
   before and after this phase's changes (§13).

Evidence standard (unchanged from Phases 1–5): a capability is claimed
only if code demonstrating it was read; a number is quoted only from a
benchmark artifact in `evaluation/results/`; absence claims are stated
as "not found by the searches in §1", with the patterns named.

---

## 2. Current AI architecture — the factual inventory

### 2.1 What exists

| Component | Reality |
|---|---|
| Detector | YOLO checkpoints via Ultralytics (PyTorch) or the Phase 4 ONNX Runtime backend. Vision only. |
| Post-processing | `ai_safety.sanitize_detections` → canonical label mapping → canonical dedup + NMS IoU 0.45 (`ModelManager.filter_airspace_detections`). Deterministic. |
| Threat posture | `ai_safety.assess_posture` — a pure function returning an enum. Civilian classes are non-hostile at any confidence; a confident drone is `requires_verification`; `confirmed_hostile` and `auto_engage_allowed` are hard-coded `False`. |
| Authorization | `ai_safety.EngagementGate` — operator ID + typed phrase `I VERIFY` + JSONL audit log. Gates opening a *simulation demo*, nothing physical. |
| Interceptor | Self-contained HTML5 canvas **game** (`interceptor_sim.py`); physics in px/s; no hardware path of any kind. |
| Text output | Templates and static strings only (inventory in §3). |
| Network at runtime | **None.** No HTTP client is imported by any runtime module. |

### 2.2 What does not exist (verified absent)

- **LLM providers / SDKs:** no OpenAI, Anthropic, Google Generative AI,
  Mistral, Cohere, Groq, Together, Ollama, LangChain, LlamaIndex,
  `transformers`, or `sentence-transformers` import, endpoint, or
  requirement anywhere. The two textual matches for "gemini" are (i)
  filenames of demo videos (`data/gemini_generated_video_*.mp4`, §4.3)
  and (ii) a developer-machine path (`.gemini/antigravity-ide/…`) inside
  two offline dataset scripts — provenance trivia, not an integration.
- **Prompts:** the word "prompt" occurs only as (a) UI copy ("engagement
  prompt", "a prompt for a human to look") and (b) the Phase 1 contract
  category `prompt_injection`, which tests *JavaScript/HTML* injection
  through the same idea — untrusted text reaching an interpreter. There
  are no system prompts, prompt templates, few-shot examples, or
  generation parameters (`temperature`, `max_tokens`, …) anywhere.
- **API keys / secrets:** no key-shaped string anywhere (pattern scan
  empty); the only `os.environ` read in the codebase is
  `AEROWATCH_TEST_PHYSICAL_CAMERA` in a test. No `.env` loading, no
  `secrets.toml` (it is gitignored), no key in git history filenames.
- **Chat / agents / tools / RAG / embeddings / memory:** no conversational
  UI, no agent loop, no function/tool calling, no vector store, no
  retrieval, no conversation state. Streamlit `session_state` holds UI
  selections and results, never dialogue.
- **OCR / captioning / VLM:** uploaded images are read only as pixels for
  the detector. No text is extracted from images, filenames are used only
  as `Path.name` (escaped, §5), and no EXIF/metadata text is consumed.
- **Tracking:** contrary to the README, there is no ByteTrack, no Kalman
  tracker and no track identity. `video_processor.BoxStabilizer` is IoU
  box smoothing; the ONNX wrapper's `track()` raises
  `NotImplementedError`. **There are no track histories for any future
  language layer to summarise — and it must never invent them.**

### 2.3 Dependencies

`requirements.txt` (pinned Phase 4): ultralytics, streamlit, opencv,
pillow, pyyaml, pandas, imageio(+ffmpeg), onnxruntime, torch, torchvision,
numpy. **No LLM-adjacent package.** Two caveats for completeness:
`huggingface_hub` and `requests` are imported by *offline dataset
builders only* (`build_*.py`, `generate_test_videos.py`) to download
datasets from Hugging Face; neither is a direct pin (they arrive
transitively via ultralytics), and neither is reachable from the running
app. The Phase 5 Commons fetcher uses stdlib `urllib` against the
Wikimedia Commons API. None of these paths sends operational data
anywhere: they fetch public datasets/media, offline, by hand.

### 2.4 Codified boundary

`tests/test_llm_boundary.py` (new, this phase) fails the suite if any
runtime module imports an LLM SDK, references an LLM endpoint or API-key
name, or if `requirements.txt` declares an LLM package. It encodes this
audit's central fact as a permanent regression gate: **an LLM can only
ever arrive through the governance process in §10, never as a drive-by
dependency.**

---

## 3. Detection-to-decision data flow (traced)

```
image/video upload or bundled sample
  → ModelManager.predict_airspace (YOLO torch | ONNX)     [raw boxes]
  → filter_airspace_detections:
        ai_safety.sanitize_detections   (finite/conf/box/class validation,
                                         per-frame cap 1000, drop reasons)
        map_detection_label             (raw class → canonical civilian
                                         Drone/Aircraft/Bird/Balloon/unknown)
        dedupe_canonical + NMS IoU 0.45 (aircraft/military_jet fix)
  → detection dicts: {box, cid, conf, internal_name, display_label,
                      hex_color, cov_pct, aspect_ratio}
  → ai_safety.assess_posture(internal_name, conf) → ThreatPosture enum
  → UI (app.py) / CLI (predict_image.py) deterministic rendering
  → [drone + confident only] EngagementGate: operator ID + "I VERIFY"
        → JSONL audit record → opens interceptor *simulation* page
```

Every arrow is typed, local and deterministic. The only untrusted-text
arrows are: uploaded **filenames**, the **operator ID / confirmation**
strings (written to the local audit log), and **class labels** coming out
of a checkpoint file (a checkpoint is user-supplied input; its declared
class names reach the UI). §5 analyses each.

### 3.1 All text generation in the system today (complete inventory)

| # | Surface | Mechanism | Grounded? |
|---|---|---|---|
| 1 | Posture banners in `app.py` ("Civilian / non-hostile…", "DRONE DETECTION — OPERATOR VERIFICATION REQUIRED… This is a model output, not a confirmed threat.") | Static templates selected by enum | Yes — written in Phase 1 to state exactly what the evidence supports |
| 2 | `predict_image.py` annotation lines (class, conf, posture, bbox) | f-string over validated fields | Yes |
| 3 | Diagnostics `dataset_summary` | Tabular counts from the filesystem | Yes |
| 4 | Engagement `summary_message` (testbench + app) | f-string template | **Was fabricated in the pre-populated results block — fixed §4.1** |
| 5 | Simulator HUD strings ("HOSTILE AIRSPACE INTRUDER", "SIM Pk 100.0% (sim)") | Static JS inside the game | In-game fiction; labelled `(sim)` in places; see §4.4 |
| 6 | README / README_AR narrative + JSON-LD schema | Human-written marketing copy | **No — see §4.2** |
| 7 | Audit log records | Structured JSON of gate events | Yes |

No other text generation exists. There are no alerts, emails, chatbots,
report writers, or "AI recommendations" in code.

---

## 4. Groundedness findings in the *existing* deterministic text

An LLM is not the only hallucination risk: templates can fabricate too.
Phase 6 found the clearest remaining example.

### 4.1 FIXED — pre-populated fabricated "certification" results (`app.py`)

On load, the interceptor page pre-populated five `MissileEngagementResult`
objects with **hard-coded** `success=True`, `miss_distance_meters=0.0`,
`max_missile_speed_mach=4.2` and
`summary_message="Kinetic intercept confirmed by …. 100% Lethal
Neutralization."`, under a button labelled "RE-RUN 100% COMBAT
CERTIFICATION SUITE" and a heading "WATCH CERTIFIED COMBAT INTERCEPTION
FOOTAGE". The identical summary string was also produced by
`video_interceptor_testbench.py` after real sim runs. These are
fabricated results presented in a results UI — precisely the failure
mode §9's schema rules exist to prevent in any future LLM layer, present
today in deterministic code. Phase 3's banned-claims test missed them
because its string list was narrower than the copy.

**Fix (this phase, minimal and presentation-only):** both summary
templates now read "Simulated intercept by … (game outcome — not a
real-world result or certification)"; the button, spinner, section
comment and footage headings say "simulation", not "certification"; and
the Phase 3 test's banned list gained `"100% Lethal Neutralization"`,
`"CERTIFIED COMBAT"` and `"100% COMBAT CERTIFICATION"` so the class of
claim, not just the instance, is guarded. No behaviour, physics, gate or
result computation changed.

### 4.2 OPEN (owner decision) — README body and AI-crawler schema

The README still contains, below the Phase 2 verification disclaimer,
unsubstantiated claims ("guaranteed 100% Kill Probability (Pk = 1.00)",
"96.7% Precision", "zero false alarms", "Mach 6.2+ … fire control
system"). Uniquely relevant to *this* phase: the README embeds a
schema.org JSON-LD block explicitly commented "Structured Schema for
Search Engines and **AI Crawlers (AEO/SEO)**" and a "GEO Entity
Definition" section — i.e. **the repository's only artifact deliberately
written to be ingested and repeated by external LLMs**, and it asserts
the very claims the project's own audits refute. An AI assistant asked
about AeroWatch today will likely parrot the schema, not the disclaimer.
Recommendation (unchanged from Phases 2–3, restated because it is now an
LLM-surface defect): either delete the false claims and the JSON-LD
description, or replace both with the measured, bounded statements from
`docs/CV_QUALITY.md` / `docs/VERIFIED_DATASET_PHASE5.md`. This is a
licensing/positioning decision for the owner; not changed unilaterally.

### 4.3 OPEN — generative-AI demo videos presented as missions

`app.py` offers two preloaded scenarios pointing at
`data/gemini_generated_video_*.mp4` — by filename, videos produced by a
generative video model — labelled "Airspace Mission 01 — Commercial
Quadcopter & Mixed Horizon" etc., with no in-UI disclosure that the
footage is AI-generated (the files are not in the repo; `data/` is
gitignored except `verified_v1/`). Whatever their origin, demo media
that a viewer could mistake for real sensor footage should be labelled
"synthetic / AI-generated demo footage" wherever it is offered. Owner
decision (label vs remove); not changed unilaterally.

### 4.4 Observation — simulator HUD language

Inside the game, the HUD unconditionally displays "HOSTILE AIRSPACE
INTRUDER" for its allowlisted game targets (`Drone/Aircraft/Bird/
Balloon/Warplane` → game archetypes), and `valPk` shows "100.0% (sim)".
Real detections can never reach the sim as a labelled civilian target —
the only gated path hard-codes the target name `"Drone"` — so this is
in-fiction signage, not a classification of a real object. It is
recorded here because any future summariser must treat simulator output
as **game state**, never as evidence about a real detection (§9 forbids
the schema from expressing it otherwise).

### 4.5 Confidence language rule (applies today and to any future layer)

YOLO `conf` is a detector score, not a calibrated probability: measured
ECE is 0.142 on the Phase 5 independent test set (0.135 on the old set).
Existing UI copy correctly says "confidence"/"model output". Any future
generated text must preserve that: **never** "there is an 85% probability
this is hostile", never convert a detection score into intent, identity
or risk percentages (full language rules in §9.3).

---

## 5. Prompt-injection & untrusted-text analysis of the *current* system

There is no prompt to inject into — but the structural analogue already
exists and was Phase 1's battleground: untrusted strings reaching the
HTML/JS interpreters of the C2 page. Current state, input by input:

| Untrusted input | Reaches | Protection today | Residual risk |
|---|---|---|---|
| Checkpoint class names (from a user-supplied `.pt`) | UI labels, sim target | Canonical mapping in `map_detection_label` (unknown → rejected/unknown), sim allowlist `validate_sim_target`, `js_literal` JSON encoding, contract cases `injection-sim-label`, `injection-script-close` pass | Low |
| Uploaded file names | Diagnostics/UI display | `Path.name` only (no directories), `html_escape` at display (`app.py` diagnostics view) | Low |
| Operator ID / confirmation text | Local JSONL audit log | Never rendered as HTML; gate compares the phrase exactly; log is local-only | Low (log content is operator-supplied by definition; treat as untrusted if ever displayed/exported — see §11) |
| Uploaded image/video bytes | Detector, ffmpeg, temp files | `ai_safety.validate_upload` (suffix + size caps), `safe_temp_upload`, ffmpeg subprocesses are list-form with timeouts; no `shell=True` anywhere (verified) | Standard media-parser risk; unchanged by this phase |
| Sample/scenario names (repo-controlled) | `unsafe_allow_html` templates | Static, repo-controlled | None |

**Structural concentration risk:** `app.py` calls
`st.markdown(…, unsafe_allow_html=True)` **66 times**. Today every
dynamic value interpolated into those strings was individually checked
(§1 method: the two dynamic cases found are escaped). This is the
single most important fact for any future LLM work: **LLM output must
never be interpolated into an `unsafe_allow_html` string.** §8 makes the
renderer the enforcement point (plain-text/`st.text` rendering or
escaped-only interpolation), because one future f-string mistake would
convert a model hallucination into stored/reflected XSS in the C2 page.

**Forward-looking injection vectors** (for the §12 threat model — none
exist today): image content via a future VLM/OCR, operator free-text
notes, filenames, video metadata, retrieved documents (RAG poisoning),
multi-turn context carry-over, and tool/function calling. §8's design
denies all of them structurally: the LLM receives only a validated JSON
context object built by code (§8.2); it has no tools; its output is
schema-validated data, rendered by deterministic code.


---

## 6. Where an LLM could legitimately help — candidate analysis

Each candidate is scored on operator value, failure consequence, and —
decisively — **whether a deterministic alternative already does the job**.
An LLM that merely paraphrases structured data adds hallucination risk
for typography; it is justified only where language itself is the task.

| # | Candidate use case | Value | Risk if wrong | Deterministic alternative | Verdict |
|---|---|---|---|---|---|
| 1 | Incident summary of one validated detection event | Medium — readable logs for handovers | High if it upgrades language ("hostile", intent) | **Template renderer over the same JSON** — already possible today, zero hallucination | **Deterministic first.** LLM polish optional, later, behind the §9 schema; never the source of facts |
| 2 | After-action / shift report from the local audit log + detection records | Medium | Medium — invented events in an official-looking record | Deterministic aggregation + fixed section templates covers counts/facts; LLM only for connective prose | Conditional — Phase 6.4 candidate, human-approved before export |
| 3 | Operator Q&A over the project's *own documentation* (audits, runbooks) | Medium — the repo now has 7 substantial docs | Low–medium — wrong procedural advice | Full-text search / index page | Conditional — read-only RAG over repo docs only, with citations; never over live detections |
| 4 | False-alarm review assistance (grouping similar FPs, drafting review notes) | Medium | Low — offline, human-verified | Clustering by class/conf/box size is deterministic | Conditional — offline only; notes are drafts, never labels |
| 5 | Annotation assistance for future datasets | Proven pattern (Phase 5 used AI grounding + human verification of every flagged image) | Medium — label noise | Human annotation | **Keep the Phase 5 pattern**: model proposes, human verifies, provenance recorded; never auto-accept |
| 6 | Multilingual operator UI (e.g. Arabic) | Medium for this user base | Low | Static i18n string tables, human-reviewed once | Deterministic — translate the *string table* offline with review; no runtime LLM |
| 7 | Free-text operator notes → structured log fields | Low–medium | Medium — silent restructuring of a record | Form fields | Reject for now; notes stay verbatim |
| 8 | Explaining *why* posture is what it is | Low | Medium — plausible-sounding wrong rationale | The posture function is 30 lines; UI can cite the exact rule that fired | Deterministic — render the rule, not a story |

**Net assessment:** there is **no current operational gap that requires
an LLM**. The strongest future case is #2/#3 (language over *records and
documents*, read-only, human-approved), and even those should ship
deterministic-first. This phase therefore designs the platform an LLM
*would* plug into — context builder, schemas, validators, harness — so
that any later adoption is a configuration of proven guardrails, not an
experiment on the safety path.

## 7. Where an LLM must never be used (prohibited, permanent)

Regardless of future approvals, an LLM must never:

1. Classify or reclassify a detection, or override/supplement the
   detector's class or confidence.
2. Determine hostility, threat level, intent, identity, ownership,
   nationality, payload, or weapon status — or phrase any of these as
   fact or probability.
3. Produce, influence, pre-fill, or expedite an authorization: no
   generated text inside the `EngagementGate` dialog, no suggested
   confirmation phrases, no summarised "case for engagement".
4. Act as a safety control, filter, or reviewer of another AI's output
   on the decision path (no LLM-as-judge over postures).
5. Call tools, execute code, browse, or trigger any action, simulation
   launch, export, or notification.
6. Receive raw images/video by default (any future VLM use is a separate
   owner decision with its own audit — §16).
7. Be a single point of failure: the system must be fully functional —
   detection, posture, gating, logging — with the LLM disabled, absent,
   or returning garbage (§8.5 degradation).
8. Generate public/marketing claims about the system's performance
   (the §4.2 failure mode, automated, would be worse).

## 8. Reference architecture for an optional future LLM layer (DESIGN ONLY)

Nothing in this section is implemented. It is the approved-shape
specification any implementation phase must follow.

### 8.1 Shape

```
validated detection/posture records (existing pipeline, unchanged)
        │  (read-only tap; LLM layer can be deleted without touching it)
        ▼
ContextBuilder (deterministic code)
  - input: SanitizedDetection + PostureAssessment + run metadata
  - allowlisted fields only; drops filenames, operator IDs, paths,
    free text; adds evidence pointers (checkpoint SHA, benchmark ref)
        ▼  JSON (schema: §9.1)
LLMGateway (provider-agnostic interface, §8.3)
  - fixed system instructions from the prompt registry (§10)
  - temperature 0, bounded max tokens, timeout, retry ≤1, budget cap
        ▼  raw model text
OutputValidator (deterministic code)
  - JSON parse + schema validation (§9.2)
  - forbidden-claim linter (word/phrase rules, §9.3)
  - fact checker: every numeric/class claim must match the context JSON
        ▼  validated structured summary (or REJECT → fallback)
Renderer (deterministic code)
  - renders validated fields as escaped plain text; never
    unsafe_allow_html with model content; visibly labelled "AI-drafted"
        ▼
Operator UI  +  AuditLogger (context hash, prompt version, model ID,
              output, validation verdict — local JSONL, §11)
```

### 8.2 Context construction rules

- The context is **built by code from typed records**, never by string
  concatenation of user input. Untrusted strings (labels, notes) enter
  only as JSON *values* in designated data fields, are length-capped,
  and are stripped of instruction-like content by the fact that the
  model is instructed — and output-validated — to treat them as data.
  Structural separation, not prompt pleading, is the control: the model
  cannot change the output schema, and the validator rejects outputs
  whose facts are not in the context.
- Fields the context may contain: canonical class, detector confidence
  (labelled "detector score"), posture enum + the rule that produced it,
  box geometry (as numbers), frame/timestamp, checkpoint filename +
  SHA-256 prefix, backend, evidence references (benchmark document +
  independent-test figures), uncertainty band.
- Fields it must **never** contain: operator ID, confirmation text,
  filenames of uploads, file paths, image bytes (per §7.6), other
  operators' data, anything from the audit log beyond the event itself.

### 8.3 Provider-agnostic gateway

- A single Python `Protocol` (`complete(system_id, context_json) ->
  str`) with the provider behind it; model ID, provider and prompt
  version are configuration, recorded on every call.
- No provider SDK types leak past the gateway; swapping providers must
  not change schemas, prompts, or validators.
- Local-model option (e.g. a small on-device model) is architecturally
  equal to a cloud provider behind the same interface — relevant because
  §11's privacy analysis favours local inference for this system.
- **Configuration is code-reviewed, not env-discovered:** enabling the
  layer requires `AEROWATCH_LLM_ENABLED=1` *and* a registry entry (§10);
  default is off; absence of a key is a normal "disabled" state, never
  an error on the detection path.

### 8.4 Structured output & rendering

- The model returns JSON only; prose outside the schema is a validation
  failure, not a display candidate.
- The renderer builds UI text from validated fields with fixed sentence
  templates — the model influences *field values within enumerations and
  bounded strings*, and the forbidden-claim linter (§9.3) plus the
  context fact-checker run before anything is shown.
- AI-drafted text is visually labelled and shown beside a "source data"
  toggle that displays the exact context JSON it was derived from.

### 8.5 Failure, timeout and degradation (fail-closed to deterministic)

| Failure | Behaviour |
|---|---|
| Provider timeout / 5xx / network loss | Deterministic template summary shown; banner "AI summary unavailable — showing standard summary". Detection path unaffected (it never waits on the LLM: summaries are async/optional). |
| Invalid JSON / schema violation / linter hit | Output discarded, event logged, deterministic summary shown. No retry loop beyond one bounded retry. |
| Budget exhausted / rate-limited | Layer disables itself until the next period; deterministic mode continues. |
| Kill switch (`AEROWATCH_LLM_ENABLED=0`) | Layer fully bypassed; zero calls attempted. |
| Model/prompt version produces eval failures (§12) | Registry entry is blocked from promotion; previous version (or deterministic mode) serves. |

### 8.6 Reliability & cost posture (design targets, not measurements)

No LLM exists, so **no latency or cost figure is claimed**. The
implementation phase must measure, on the evaluation harness, before
any pilot: p50/p95 summary latency, cost per 1,000 summaries at the
chosen provider's then-current prices, and failure rates. Design
ceilings to enforce in code: per-call timeout 10 s, max output tokens
≤ 400 for summaries, per-day call budget configurable with a hard
default, and caching keyed by context hash so identical events are
never re-summarised. Any figure quoted in a future report must come from
those measurements, in the Phase 2/5 tradition of separating measured
from claimed.

## 9. Schemas and language rules (DESIGN ONLY)

### 9.1 Input context schema (summary)

`evaluation/llm_schemas.json` (added this phase) defines
`IncidentContext` v1: `schema_version`, `event_id` (hash), `checkpoint`
{name, sha256_prefix}, `backend` enum, `detections[]` {canonical_class
enum, detector_score 0–1, posture enum, posture_rule, box_xyxy,
size_band enum}, `frame` {width, height, timestamp_iso}, `evidence`
{benchmark_doc, independent_test: {mAP50, recall} — the Phase 5 figures},
`uncertainty_band` enum. No free-text fields except none at all in v1 —
deliberately: the first version has **no untrusted-string input**.

### 9.2 Output schema

`IncidentSummary` v1: `headline` (≤120 chars), `facts[]` — each fact is
{`field_ref` (a JSON pointer into the context), `statement` ≤200
chars} — `uncertainties[]` (strings from an enum-seeded list),
`recommended_operator_action` — **enumeration only**:
`none | visually_verify | review_recording | consult_documentation`,
`limitations` (must include the fixed sentence "A model classification
is not a confirmation of identity, intent or hostility."). Free
recommendation text is not in the schema, by design.

### 9.3 Forbidden-claim linter rules (validator, deterministic)

Reject any output that: names a posture other than the context's;
contains hostility/intent/identity vocabulary (`hostile`, `enemy`,
`attacker`, `threat confirmed`, `weapon`, nationality or organisation
names, `armed`, `payload`); converts a detector score into a percentage
probability of anything except the score itself; states an action was
taken or authorised; invents numbers, times, distances, speeds, or
track histories not present in the context (every numeral in the output
must appear in, or be arithmetically derived by the *renderer* from,
the context); or claims certification/verification the evidence fields
do not support.

## 10. Prompt governance (DESIGN ONLY)

- **Registry, not strings-in-code:** prompts live as versioned files
  under `prompts/` (`<id>.v<major>.<minor>.md` with YAML front-matter:
  id, version, owner, purpose, approved_by, date, eval_suite_version).
  Code references prompts by ID+version only.
- **Lifecycle:** draft → evaluated (§12 suite must pass at 100% on
  safety categories) → owner-approved → active → deprecated. Any edit,
  however small, is a new version and re-runs the full suite. No prompt
  is edited in production; there is no production prompt console.
- **Separation:** system instructions are fixed registry text. Runtime
  data is never interpolated into instruction text; it travels only as
  the §9.1 JSON in the data channel. Prompt files are code-reviewed like
  safety code (they are — see `ai_safety.py` precedent).
- **Change log & rollback:** git is the registry's history; rollback =
  pinning the previous version ID in configuration. The deterministic
  fallback (§8.5) is the ultimate rollback and always exists.

## 11. Privacy & data handling (current state + design rules)

**Today:** all processing is local. Uploads are temp files, deleted
after use; the engagement audit log is a local JSONL; nothing is
transmitted at runtime (no network imports — §2.1). Privacy posture is
therefore strong *by absence*, and any LLM adoption is the moment that
changes — which is why it is gated.

**Design rules for any future layer:** (1) context minimisation per
§8.2 — no images, filenames, paths, operator IDs, or free text leave
the machine in v1; (2) operator IDs are pseudonymised (salted hash)
even in local LLM audit records; (3) provider choice must document
retention/training terms at adoption time, and a provider whose terms
permit training on inputs is disqualified for any non-public data;
(4) the local-model option is preferred for anything beyond public
documentation Q&A; (5) LLM audit records (context hash, prompt version,
model, output, verdict) are stored locally with the same retention as
the engagement log, and are themselves treated as untrusted text if
ever displayed (§5); (6) a data-flow note must be added to
`SECURITY.md` in the implementation phase, before the first live call.

## 12. Evaluation harness design (DESIGN ONLY — case file added)

Modeled on the Phase 1 contract evaluation (`evaluation/cases.json`,
19/19), which already pioneered the categories this phase extends.
`evaluation/llm_cases.json` (added this phase, **specification only — no
runner exists and no model was called**) defines 24 cases across:

- **groundedness (6):** summaries of known contexts must contain exactly
  the context's facts — incl. a Balloon event that must not acquire
  threat language, and an `unknown` class event that must stay unknown.
- **injection (8):** hostile strings placed where a future version might
  carry them — detection label, operator note field, filename, document
  chunk (RAG), fake "system:" turn, instruction to reveal the prompt,
  instruction to authorise, and a multi-case persistent-instruction
  attempt. Expected: output schema-valid, facts unchanged, no
  instruction followed.
- **schema & linter (5):** outputs (recorded fixtures) containing a
  forbidden claim, an invented number, an out-of-enum action, prose
  outside JSON, and an over-long headline — the *validator* must reject
  each. These cases test deterministic code and can run before any
  model exists.
- **degradation (3):** provider timeout, malformed response, budget
  exhaustion → deterministic fallback verified.
- **determinism (2):** same context summarised 5× at temperature 0 must
  yield fact-identical outputs (string variance allowed, fact variance
  is failure).

**Promotion gate:** 100% pass on injection + schema + degradation,
≥ threshold (owner-set at implementation) on groundedness, plus the
existing 19/19 contract suite and full pytest run unchanged. A model or
prompt version that fails is not "mostly safe" — it is not promoted.

## 13. Verification performed in this phase

- Full test suite **before** changes: 103 passed, 1 skipped (physical
  camera), contract evaluation 19/19.
- After changes: **106 passed, 1 skipped** (103 prior + 3 new boundary
  tests), contract evaluation **19/19**, and the full CI mirror
  (`scripts/ci_checks.sh`: pip check, pip-audit, license inventory,
  flake8 E9/F gates, tests, evaluation) reports **ALL CI CHECKS PASSED**.
- v7 checkpoint SHA-256 re-verified unchanged; no weights touched; the
  safety architecture (`ai_safety.py`) is byte-unchanged this phase.

## 14. What was changed in this phase (complete list)

1. `app.py`, `video_interceptor_testbench.py` — fabricated-claims fix
   (§4.1), presentation strings only.
2. `tests/test_phase3_audits.py` — banned-claims list extended (§4.1).
3. `tests/test_llm_boundary.py` — new no-LLM regression guard (§2.4).
4. `docs/LLM_PROMPT_AUDIT_PHASE6.md` — this document.
5. `evaluation/llm_cases.json`, `evaluation/llm_schemas.json` — harness
   and schema **specifications** (§9, §12); no runner, no model calls.
6. `docs/AI_ENGINEERING.md` — one pointer paragraph to this audit.

**Explicitly not done:** no LLM implemented or called; no provider
account, key, or spend; no prompt written for production use; no change
to detection, posture, gating, simulation behaviour, weights, datasets,
licensing, or deployment; no push to `main`; no PR.

## 15. Roadmap (each step separately owner-gated)

| Step | Content | Gate to proceed |
|---|---|---|
| 6.1 (this phase) | Audit + design + boundary guard | — done |
| 6.2 | Implement the *deterministic* parts only: context builder, schemas, validator + linter, harness runner with recorded fixtures (still no live model) | Owner approves design |
| 6.3 | Deterministic template summariser shipped (no LLM) — captures most of use-case #1's value at zero risk | 6.2 harness green |
| 6.4 | Optional: one provider behind the gateway, **shadow mode** (outputs logged, never shown), evaluated against §12 | Owner picks provider + budget (§16) |
| 6.5 | Limited pilot: AI-drafted summaries visible, labelled, read-only, kill switch on | Shadow-mode eval ≥ gate; owner approval |
| Never | LLM on the classification/posture/authorization path (§7) | — |

## 16. Decisions required from the owner

1. **Adopt an LLM at all?** This audit's recommendation: not now —
   ship 6.2/6.3 (deterministic) if summaries are wanted; revisit 6.4
   only with a concrete operator need.
2. If yes later: **provider & data posture** — local model vs cloud;
   §11 disqualifies training-on-inputs terms for non-public data.
3. **Images:** confirm the §7.6 default — raw imagery never leaves the
   machine — or commission a separate VLM audit.
4. **Budget ceiling** for any cloud pilot (calls/day, max tokens) —
   required before 6.4 by §8.6.
5. **README claims & JSON-LD (§4.2):** correct, qualify, or remove —
   pending since Phase 2; now also an external-LLM misinformation
   surface.
6. **Gemini-generated demo videos (§4.3):** label as AI-generated in the
   UI, or remove the presets.

## 17. Final verdicts

| Question the phase set out to answer | Answer |
|---|---|
| Does AeroWatch use an LLM or prompts today? | **No — verified exhaustively; the premise does not apply to this codebase.** |
| Is its current AI language output safe & grounded? | Mostly — deterministic and conservative by design (Phase 1); one fabrication block found and fixed; README/schema claims remain the owner's open decision. |
| Could an LLM help? | Marginally, later, for prose over records/docs — after deterministic alternatives, behind schemas, validators, a harness and a kill switch. |
| May an LLM touch detection, posture or authorization? | **Never (§7).** |
| Is anything implemented or deployed by this phase? | No model, no prompt, no provider — audit, design, two presentation fixes, and guards only. |
