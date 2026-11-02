# Contributing to AeroWatchAI

Thanks for your interest. This is a research-grade computer-vision
project with an unusually strict honesty bar: claims must be
measured, and safety behaviour is contractual. Please read this
before opening a pull request.

## Environment setup

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

```bash
# Linux/macOS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
python preflight.py
```

Python 3.12 with the pinned lockfile is the verified environment.

## Before you submit

Run the full battery — all of it must pass:

```bash
python -m pytest tests/ -q                 # full suite (169+ tests)
python evaluation/run_evaluation.py        # safety contract: 19/19
bash scripts/ci_checks.sh                  # local CI mirror: ALL CHECKS PASSED
python -m flake8 . --count --select=E9,F63,F7,F82 --statistics
```

For UI changes, also verify the affected pages render (Streamlit
AppTest or a real browser) and say how you verified in the PR.

## Non-negotiable rules

1. **AI safety is a contract.** Never weaken `ai_safety.py`
   semantics: civilian classes stay non-hostile, `confirmed_hostile`
   and `auto_engage_allowed` stay hard-coded `False`, the engagement
   gate keeps requiring operator ID + typed phrase + audit logging.
   Safety changes need new tests proving the contract still holds.
2. **No invented capabilities or numbers.** Performance figures come
   from committed benchmark artifacts on the frozen verified test
   set (`data/verified_v1`), or they don't ship. Never reintroduce
   withdrawn claims (see Phase 7 guards in
   `tests/test_product_integrity_phase7.py`).
3. **No LLM integration** without an explicit maintainer decision —
   a regression test enforces the current boundary.
4. **Dataset licensing.** Training/evaluation data must have a
   verifiable licence recorded per file (the `verified_v1` manifest
   is the model). Unverifiable-licence data is not accepted.
5. **History privacy.** No media bytes, full paths, or operator
   identifiers in the history database. Ever.
6. **Licensing honesty.** Do not change `LICENSE`, add license
   badges, or describe the project as commercially cleared — the
   licensing position is documented in
   `docs/LICENSING_DECISION.md` and is the maintainer's decision.

## Pull requests

- One topic per PR; describe the change, the evidence, and the
  tests you ran (paste the summary lines).
- Bug reports and fixes should include a reproduction: input,
  expected, actual, environment (`preflight.py` output helps).
- Model changes require benchmark results on the frozen test set
  with the exact command used.

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
