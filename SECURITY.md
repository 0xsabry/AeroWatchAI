# Security Policy — AeroWatch AI

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.6.x   | :white_check_mark: |
| 2.5.x   | :white_check_mark: |
| < 2.5   | :x:                |

## Reporting a Vulnerability

The AeroWatch AI project takes security and operational integrity seriously. If you discover a security vulnerability, buffer exploit, model deserialization issue, or data integrity concern:

1. **Do not disclose the issue publicly** in GitHub issues or discussions.
2. Please notify the lead author and maintainer (**Eng. Mohamed Sabry**) via GitHub security advisories or direct secure correspondence.
3. Include a description of the vulnerability, affected components, steps to reproduce, and potential impact.
4. We aim to acknowledge receipt of vulnerabilities within 48 hours and provide a resolution or patch timeline promptly.

Thank you for helping keep the airspace defense software ecosystem safe.

## AI Safety Boundaries (added with ai_safety.py)

- **Model output is untrusted.** All detections pass the contract in
  `ai_safety.sanitize_detections` before rendering, metrics, or any
  downstream decision. Malformed output is dropped and counted, never
  silently trusted.
- **Classification is not hostility.** Civilian classes (Aircraft, Bird,
  Balloon) are never presented as hostile and never receive an engagement
  prompt. A drone detection only ever means "operator verification
  required". The system has no automatic engagement path.
- **Simulation launch is gated.** Opening the interceptor simulation from a
  detection requires an operator ID and a typed confirmation, and is written
  to `runs/engagement_audit.jsonl` (no image data or credentials are logged).
- **Checkpoints:** `.pt` files are PyTorch pickle archives — loading one is
  code execution. Only load checkpoints from trusted sources, and verify the
  SHA-256 fingerprint that `ModelManager` logs at load time against a known
  value before operational use.
- **Uploads** are extension- and size-limited (images 20 MiB, videos
  200 MiB) and written to temporary files that are cleaned up after use.
- **No authentication** is built into the Streamlit app. Do not expose it to
  untrusted networks without an authenticating reverse proxy.
