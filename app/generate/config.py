"""
config.py
---------
Single source of truth for all generate stage settings.

No logic lives here — only paths, constants, and environment variables.
Every other file in this package imports from here.

Pipeline stages that use this file:
    generator.py — Groq client, model, streaming, prompt loading
    run.py       — combined retrieve + generate test runner
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR     = Path(__file__).parent
PROMPTS_FILE = BASE_DIR / "prompts.yaml"


# =============================================================================
# LLM — Groq
# =============================================================================
# llama-3.3-70b-versatile chosen for strong instruction-following at low
# latency/cost — already validated in the v1 pipeline.
#
# Fallback models are tried in order if the primary model errors or is
# rate-limited, so a transient Groq issue doesn't take down generation.

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

GROQ_FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

# Generation parameters
TEMPERATURE      = 0.1   # low — we want grounded, consistent answers, not creativity
MAX_TOKENS       = 1024  # generous for detailed policy answers, not unbounded
STREAM_RESPONSE  = True  # token-by-token streaming to the caller


# =============================================================================
# PROMPT VERSIONING
# =============================================================================
# Prompts live in prompts.yaml, not hardcoded in Python. This lets us
# iterate on prompt wording without touching code, and keep a history of
# what changed and why (every version has a "notes" field for that).
#
# ACTIVE_PROMPT_VERSION controls which version is used at runtime — bump
# this after testing a new version, rather than deleting old ones.

ACTIVE_PROMPT_VERSION = "v1"


# =============================================================================
# RETRIEVAL INTEGRATION
# =============================================================================
# How many chunks to request from app.retrieve.run.search() per query.
# Matches RERANK_TOP_K in app/retrieve/config.py — kept as a separate
# constant here so generate/ can override it independently if needed
# (e.g. fewer chunks for faster responses, more for complex questions).

CONTEXT_CHUNK_COUNT = 5

# Maximum characters of context to send to the LLM per chunk — guards
# against a single unusually large chunk blowing the context budget.
MAX_CHUNK_CHARS = 2000


# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


# =============================================================================
# VALIDATION
# =============================================================================

def validate() -> None:
    """
    Check required environment variables and that prompts.yaml exists
    with the active version defined. Called at startup by run.py and
    each stage's __main__.
    """
    errors = []

    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set in .env")

    if not PROMPTS_FILE.exists():
        errors.append(f"prompts.yaml not found at {PROMPTS_FILE}")
    else:
        import yaml
        with open(PROMPTS_FILE, encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        if ACTIVE_PROMPT_VERSION not in prompts.get("versions", {}):
            errors.append(
                f"ACTIVE_PROMPT_VERSION='{ACTIVE_PROMPT_VERSION}' "
                f"not found in {PROMPTS_FILE}"
            )

    if errors:
        print("\n❌  Generate config errors:")
        for e in errors:
            print(f"    → {e}")
        raise SystemExit(1)

    print("\n✅  Generate config validated")
    print(f"    Model:           {GROQ_MODEL}")
    print(f"    Fallbacks:       {GROQ_FALLBACK_MODELS}")
    print(f"    Temperature:     {TEMPERATURE}")
    print(f"    Streaming:       {STREAM_RESPONSE}")
    print(f"    Prompt version:  {ACTIVE_PROMPT_VERSION}")
    print(f"    Context chunks:  {CONTEXT_CHUNK_COUNT}")


if __name__ == "__main__":
    validate()
