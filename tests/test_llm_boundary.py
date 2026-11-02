"""Phase 6 boundary guard: the runtime contains no LLM integration.

The Phase 6 audit (docs/LLM_PROMPT_AUDIT_PHASE6.md) verified that
AeroWatch has no LLM provider, SDK, prompt layer, API key or external
AI API, and designed the only permitted future shape of one. This test
codifies that audited state: if an LLM dependency ever appears in a
runtime module, the suite fails and the change must go through the
Phase 6 governance process (prompt registry, schema validation,
evaluation harness, owner approval) instead of landing silently.

Dataset builders (build_*.py, evaluation/fetch_*.py) are out of scope:
they run offline, download data, and never feed an LLM. Documentation
is out of scope: it *discusses* LLMs.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNTIME_MODULES = [
    "app.py",
    "ai_safety.py",
    "model_manager.py",
    "onnx_backend.py",
    "video_processor.py",
    "predict_image.py",
    "interceptor_sim.py",
    "video_interceptor_testbench.py",
    "camera_manager.py",
    "diagnostics.py",
]

FORBIDDEN_IMPORTS = (
    "openai",
    "anthropic",
    "langchain",
    "llama_index",
    "transformers",
    "ollama",
    "cohere",
    "mistralai",
    "google.generativeai",
    "google.genai",
    "sentence_transformers",
    "groq",
    "together",
)

FORBIDDEN_TOKENS = (
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "system_prompt",
    "chat.completions",
)


def test_no_llm_imports_in_runtime_modules():
    for rel in RUNTIME_MODULES:
        text = (ROOT / rel).read_text()
        for name in FORBIDDEN_IMPORTS:
            assert f"import {name}" not in text, f"{rel} imports {name}"
            assert f"from {name}" not in text, f"{rel} imports {name}"


def test_no_llm_endpoints_or_keys_in_runtime_modules():
    for rel in RUNTIME_MODULES:
        text = (ROOT / rel).read_text()
        for token in FORBIDDEN_TOKENS:
            assert token not in text, f"{rel} contains {token!r}"


def test_requirements_declare_no_llm_packages():
    reqs = (ROOT / "requirements.txt").read_text().lower()
    for pkg in ("openai", "anthropic", "langchain", "transformers",
                "ollama", "cohere", "mistralai", "groq", "together"):
        assert pkg not in reqs, f"requirements.txt declares {pkg}"
