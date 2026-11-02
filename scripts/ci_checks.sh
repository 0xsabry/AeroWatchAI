#!/usr/bin/env bash
# AeroWatch AI — local mirror of the CI pipeline (.github/workflows/ci.yml).
# Run from the repo root inside the project venv:  bash scripts/ci_checks.sh
# Exits non-zero at the first failing gate, exactly like CI.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "== lockfile consistency =="
pip check

echo "== dependency vulnerability scan (pip-audit) =="
# CI upgrades setuptools before this step; do the same locally if your
# env's build tooling is flagged:  pip install --upgrade setuptools
pip-audit --progress-spinner off

echo "== license inventory validation =="
python evaluation/license_inventory.py --out /tmp/licenses_inventory_ci.json > /dev/null
python - << 'PYEOF'
import json
inv = json.load(open('/tmp/licenses_inventory_ci.json'))
names = {d['name'].lower().replace('_', '-') for d in inv['distributions']}
required = {'ultralytics', 'streamlit', 'opencv-python', 'pillow', 'pyyaml', 'pandas', 'imageio', 'imageio-ffmpeg'}
missing = required - names
assert not missing, f'license inventory missing: {missing}'
ultra = next(d for d in inv['distributions'] if d['name'] == 'ultralytics')
assert 'AGPL' in ultra['license_declared'], 'ultralytics AGPL not recorded'
print('license inventory OK:', inv['count'], 'distributions')
PYEOF

echo "== lint (CI selection) =="
python -m flake8 app.py ai_safety.py onnx_backend.py model_manager.py interceptor_sim.py video_interceptor_testbench.py camera_manager.py video_processor.py diagnostics.py predict_image.py evaluate_videos.py --count --select=E9,F63,F7,F82 --statistics

echo "== import smoke =="
python -c "import interceptor_sim; import video_interceptor_testbench; import model_manager; import onnx_backend; print('ALL CORE MODULES IMPORTED SUCCESSFULLY')"

echo "== test suite (incl. ONNX parity gates) =="
python -m pytest tests/ -q -k "not test_physical_webcam"

echo "== AI contract evaluation =="
python evaluation/run_evaluation.py

echo "ALL CI CHECKS PASSED"
