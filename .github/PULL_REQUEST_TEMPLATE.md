## What changed

<!-- One topic per PR. Describe the change and why. -->

## Evidence

<!-- Benchmarks, screenshots, or artifacts supporting the change.
     Performance claims must cite a committed benchmark artifact. -->

## Verification

- [ ] `python -m pytest tests/ -q` — result:
- [ ] `python evaluation/run_evaluation.py` — 19/19
- [ ] `bash scripts/ci_checks.sh` — ALL CHECKS PASSED
- [ ] UI changes verified in AppTest / real browser (describe):

## Safety & licensing checklist

- [ ] `ai_safety.py` contract unchanged, or new tests prove it still holds
- [ ] No withdrawn claims reintroduced; no invented metrics
- [ ] No LLM added; no new network calls at runtime
- [ ] No media bytes / full paths / operator IDs added to history storage
- [ ] No LICENSE or licensing-position changes
- [ ] Checkpoint files unchanged (v7 SHA-256 `992080af…`)
