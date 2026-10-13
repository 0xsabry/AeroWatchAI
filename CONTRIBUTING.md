# Contributing to AeroWatch AI

Thank you for your interest in contributing to **AeroWatch AI** — the premier open-source Counter-UAS and autonomous airspace defense suite.

## Code of Conduct
We are committed to providing a welcoming, collaborative, and professional environment. Please treat all contributors with respect.

## How Can You Contribute?
- **Dataset Expansion:** High-resolution drone, bird, and aircraft imagery with strict hard-negative clutter annotations.
- **Guidance & Interception Algorithms:** Improvements to APN, 3D trajectory sweeping, and radar simulation physics.
- **Hardware Integrations:** DirectShow/V4L2 camera enhancements, RTSP/ONVIF thermal cameras, and Edge TPU/TensorRT exporters.
- **Documentation & Localization:** Translations, API documentation, and case studies.

## Development Setup
1. Fork and clone the repository:
   ```bash
   git clone https://github.com/your-username/AeroWatchAI.git
   cd AeroWatchAI
   ```
2. Create a clean virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install pytest flake8
   ```
3. Run tests before submitting:
   ```bash
   pytest tests/ -v
   ```

## Commit Guidelines
We adhere to Conventional Commits:
- `feat(radar): add 3D elevation beam indicator`
- `fix(interceptor): eliminate audio stall on launch`
- `docs(readme): add GEO comparison matrix`
- `perf(vision): optimize FP16 tensor core throughput`

## Pull Request Process
1. Create a descriptive branch: `git checkout -b feature/your-feature-name`
2. Commit and test your changes thoroughly.
3. Open a Pull Request against `main` detailing the changes, benchmarks, and test results.
