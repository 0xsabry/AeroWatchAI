# AeroWatch AI — License & IP Audit (Phase 3)

**This document is an engineering evidence record, not legal advice.**
It separates (a) verified facts from primary sources, (b) positions
published by upstream vendors, and (c) genuinely open questions that
require the project owner's decision and, for anything commercial,
professional legal counsel. No license was changed by this audit; the
repository remains labelled MIT by its author, and §6 lists exactly
why that label is currently **not sufficient** to describe the
project's actual licensing situation.

Evidence collected 2026-11-02. Machine-readable inventory:
`docs/licenses_inventory.json` (regenerate with
`evaluation/license_inventory.py`). Checkpoint fingerprints:
`docs/CV_QUALITY.md` / Phase 1 records.

## 1. The central fact

The project is a derivative work built on **Ultralytics YOLO**:

- The installed `ultralytics` package (8.4.198) declares
  **License: AGPL-3.0** in its own metadata, carries the classifier
  "GNU Affero General Public License v3 or later (AGPLv3+)", and
  ships the full AGPL-3.0 text in its dist-info. *(Verified from the
  package itself.)*
- Every custom checkpoint in `weights/` was trained with Ultralytics
  (training scripts in-repo), and the runtime imports Ultralytics for
  all inference.
- The repository's own `LICENSE` is MIT, and the word "AGPL" appeared
  nowhere in the repo before Phase 2/3 documentation.

**Ultralytics' published position** (ultralytics.com/license,
accessed 2026-11-02): using Ultralytics code *or models trained with
it* requires either (a) open-sourcing the **entire project** under
AGPL-3.0 — "including models you train", applying "even if you use
YOLO only internally or for R&D" and to SaaS/API deployment — or
(b) an **Ultralytics Enterprise License** for proprietary/commercial
use. Whether a court would sustain the full breadth of that position
(specifically, AGPL's reach into trained model weights as opposed to
the software) is a legal question this audit does not decide; what is
not in doubt is that (i) the *software* dependency is AGPL-3.0, which
has real copyleft obligations on distribution and network use of the
combined work, and (ii) the vendor asserts the broader position and
sells the alternative. Both facts belong in any distribution
decision.

## 2. Software dependency inventory (direct + key transitive)

From `requirements.txt` (which pins **nothing** — all `>=` minimums;
that is itself a reproducibility/supply-chain finding) and the active
environment's package metadata:

| Package | Version here | Declared license (evidence) | Notes |
|---|---|---|---|
| ultralytics | 8.4.198 | **AGPL-3.0** (metadata + LICENSE file) | The binding constraint; see §1 |
| torch | 2.5.1+cpu (repo comment references 2.14.1+cu132 elsewhere — environments differ) | BSD-3-Clause (metadata + LICENSE/NOTICE) | Permissive |
| torchvision | 0.20.1+cpu | BSD (LICENSE file) | Permissive |
| opencv-python | 5.0.0.93 | Apache-2.0 (metadata + LICENSE) | Permissive; bundles third-party components under cv2/LICENSE-3RD-PARTY.txt |
| streamlit | 1.66.1 | Not present in local metadata snapshot; upstream project is Apache-2.0 — **verify at distribution time** | Permissive upstream |
| numpy | 2.5.4 | LICENSE.txt ships in dist-info (BSD-style, plus bundled-component licenses) | Permissive |
| pandas | 3.1.0 | LICENSE ships in dist-info (BSD-3-Clause) | Permissive |
| pillow | 13.0.0 | LICENSE ships in dist-info (HPND/MIT-style) | Permissive |
| PyYAML | 6.0.3 | MIT | Permissive |
| imageio / imageio-ffmpeg | 2.38.1 / 0.7.0 | BSD-2-Clause | imageio-ffmpeg **bundles an ffmpeg binary** (LGPL/GPL components depending on build) — flag for any binary redistribution |
| setuptools (build tooling) | 78.1.0 | — | Only package with known vulnerabilities in this env (see PRODUCT_READINESS) |

Full machine-readable record: `docs/licenses_inventory.json`.
No bundled third-party JS/CSS/fonts were found beyond what Streamlit
and the project's own hand-written simulation ship; the simulation
code is original to this repo.

## 3. Model checkpoint licensing

| Checkpoint | SHA-256 (first 12) | Origin | Base / lineage | License position |
|---|---|---|---|---|
| aerowatch_v7_balanced.pt (= best.pt, byte-identical) | `992080afb8f3…` | Custom-trained in this project (`train_v7_balanced.py`, 30 ep, real_aerial_v4_balanced) | From clean Ultralytics `yolo26n.pt` COCO pretrained | Trained with AGPL software from AGPL pretrained weights, on datasets with **undeclared licenses** (§4). Cannot be asserted MIT. Distribution rights: **unresolved** |
| aerowatch_v6_expert.pt | `f4e441652903…` | Custom-trained (`train_v6_expert.py`, real_aerial_v3) | From v5_pro (fallback real_v1) | Same, plus lineage through undocumented checkpoints |
| aerowatch_v5_pro.pt | `4ebcf64148aa…` | **No training script exists** — provenance undocumented | Unknown | **Unknown** — do not distribute |
| aerowatch_real_v1.pt | `b6a38e2ffbe1…` | **No training script exists** — provenance undocumented | Unknown | **Unknown** — do not distribute |
| yolo26n.pt (repo root + weights/) | `9b09cc8bf347…` | Official Ultralytics COCO pretrained | Ultralytics release | AGPL-3.0 per Ultralytics (weights), COCO annotations CC-BY 4.0 layer for its training data |

Missing information that blocks a definitive answer for the custom
checkpoints: (1) the training datasets' actual licenses (§4);
(2) a legal determination of AGPL's application to trained weights;
(3) any record of the v5/real_v1 training runs.

## 4. Dataset licensing

Verified via the Hugging Face API on 2026-11-02 (full detail in
`docs/DATASET_PROVENANCE.md` §1/§3):

- **Rigoleto/drone-detection** (rev b8ad8fc3…): **no license declared**
  on HF (no cardData license, no license tag).
- **matisdsp/drone-bird-detection-dataset** (rev 5f6c2789…): **no
  license declared** on HF. Roboflow-export filenames suggest an
  upstream Roboflow project whose terms were not recorded.
- **a2015003713/military-aircraft-detection-dataset** (rev
  0cb9ddd0…): **no license declared** on HF.
- The eval manifest's claims (CC-BY 4.0 / CC-BY 2.0 / CC0) are
  **unverified against any upstream source** and may reflect original
  publisher pages that were never recorded. "Authentic Drone Flight
  Capture" and "Civilian Balloon Dataset" have no recoverable source
  at all.
- Consequence: the project currently **cannot demonstrate a right to
  redistribute the training data, and cannot demonstrate that the
  trained weights are free of dataset-imposed conditions**. Public
  accessibility on HF is not a license grant.

## 5. Distribution scenarios

Engineering reading of the facts above — **not legal advice**:

| # | Scenario | What the evidence supports | Open questions |
|---|---|---|---|
| 1 | Personal local use | No distribution occurs; AGPL obligations are not triggered by private use. Lowest risk. | None material |
| 2 | Internal organizational use | Still no external distribution, but Ultralytics' published position explicitly claims internal/R&D use of models requires AGPL compliance or Enterprise. AGPL's text ties obligations to conveying/network access by outside users. | Whether internal multi-user access counts as network interaction under §13 — legal advice territory |
| 3 | Public open-source distribution | Viable **only** if the whole combined work is offered under AGPL-3.0 (relicensing the project's own code from MIT → AGPL, including weights per vendor position) AND dataset rights are resolved. MIT-labelled distribution of the current bundle would misdescribe the AGPL component. | Owner decision: relicense, or remove/replace Ultralytics. Dataset rights per §4 |
| 4 | Commercial desktop application | Not supportable today: proprietary distribution conflicts with AGPL absent an Enterprise license; dataset licenses undeclared. | Enterprise license purchase (owner decision, paid), or re-platform to a permissively licensed detector + clean data |
| 5 | Hosted online service (public SaaS) | AGPL §13 (network clause) is squarely implicated for the software; vendor position claims Enterprise required for proprietary SaaS. | Same as 4, plus service-level source-offer mechanics if AGPL route chosen |
| 6 | Private commercial SaaS | Same AGPL analysis as 5 — "private" deployment does not remove §13 source-offer duties to users under the AGPL reading, and the vendor explicitly claims this case. | Legal counsel + owner decision |

## 6. Decisions requiring the owner's approval

1. **Licensing route for the project itself:** (a) relicense the
   whole project (code + weights) to AGPL-3.0 and resolve dataset
   rights, keeping Ultralytics; (b) purchase an Ultralytics
   Enterprise License (paid; not purchased by this audit);
   (c) re-platform inference/training to a permissively licensed
   stack (e.g. a non-Ultralytics detector) — an engineering project,
   not a paperwork change; or (d) restrict the project to personal /
   research use under the current labels with the caveats in these
   docs. *This audit recommends (d) as the interim state and a
   deliberate choice among (a)–(c) before any distribution.*
2. **v5_pro / real_v1:** keep as undocumented local artifacts or
   remove from the repo; they should not be distributed in any case.
3. **Dataset rights:** invest in recovering/replacing the three HF
   sources with license-verified data (see DATASET_PROVENANCE §4), or
   accept that weights trained on them stay undistributable.
4. **README/license labelling:** whether to amend the MIT labelling
   now (e.g. "code MIT; models and combined work subject to
   AGPL-3.0 — see LICENSE_AUDIT") ahead of decision 1. The README's
   verification note already links here; the `LICENSE` file itself
   was **not** modified by this audit.

## 7. What this audit did NOT do

- Did not change any license, copyright notice, or the `LICENSE` file.
- Did not purchase or agree to anything.
- Did not treat the scanner output (`licenses_inventory.json`) as a
  legal conclusion — it is declared-metadata evidence only; five
  packages declare no License field at all and were resolved from
  shipped LICENSE files or flagged as unverified (streamlit).
- Dependency *versions* above describe the audit environment;
  `requirements.txt` is unpinned, so a fresh install may resolve
  different versions with different metadata. Pin before relying on
  this inventory for a release.
