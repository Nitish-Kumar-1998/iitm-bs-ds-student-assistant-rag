"""
generator.py
------------
Stage 3 of the RAG pipeline: turns retrieved chunks + a question into a
grounded, cited, streamed answer via Groq.

INPUT:  query string + list of chunks (from app.retrieve.run.search())
OUTPUT: streamed text response, grounded strictly in the provided chunks

Pipeline:
    1. Load the active prompt version from prompts.yaml
    2. Format retrieved chunks into a numbered context block
    3. Fill the prompt template with that context
    4. Call Groq (streaming) with the system prompt + user question
    5. Yield tokens as they arrive (for streaming) or return full text

Model fallback:
    If the primary model (GROQ_MODEL) errors or is rate-limited, we retry
    with each model in GROQ_FALLBACK_MODELS in order before giving up.
    This keeps the assistant available even during a Groq outage on one
    specific model.

Run standalone (interactive test, requires retrieve stage working):
    python -m app.generate.generator "what is the OPPE exam"
"""

import sys
import logging

import yaml
from groq import Groq

from app.generate.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_FALLBACK_MODELS,
    TEMPERATURE,
    MAX_TOKENS,
    STREAM_RESPONSE,
    PROMPTS_FILE,
    ACTIVE_PROMPT_VERSION,
    MAX_CHUNK_CHARS,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("generate.generator")

# Module-level singletons — created once, reused across calls
_groq_client: Groq | None = None
_prompts_cache: dict | None = None


# =============================================================================
# GROQ CLIENT (singleton)
# =============================================================================

def get_groq_client() -> Groq:
    """Return a shared Groq client instance, created once and reused."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialised")
    return _groq_client


# =============================================================================
# PROMPT LOADING
# =============================================================================

def load_prompt(version: str = ACTIVE_PROMPT_VERSION) -> str:
    """
    Load a system prompt template from prompts.yaml by version.

    Cached after first load — prompts.yaml is small and doesn't change
    at runtime, so re-reading it on every call would be wasted I/O.

    Args:
        version — key under "versions:" in prompts.yaml (e.g. "v1")

    Returns:
        The raw "system" template string, containing a {context}
        placeholder to be filled by build_context().

    Raises:
        ValueError if the requested version doesn't exist in the file.
    """
    global _prompts_cache
    if _prompts_cache is None:
        with open(PROMPTS_FILE, encoding="utf-8") as f:
            _prompts_cache = yaml.safe_load(f)

    versions = _prompts_cache.get("versions", {})
    if version not in versions:
        available = list(versions.keys())
        raise ValueError(
            f"Prompt version '{version}' not found. Available: {available}"
        )

    return versions[version]["system"]


# =============================================================================
# CONTEXT BUILDER
# =============================================================================

def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Each chunk becomes a numbered source the model can cite back, e.g.:

        [1] Source: OPPE System Compatibility Test - RULES
            Section: BS-DS May 2026 Grading Document (Student) > OPPE System Compatibility Test - RULES
            Content: In order to ensure that your system (laptop) is...

    Args:
        chunks — list of dicts from app.retrieve.run.search(), each with
                 a "payload" containing content, doc_title, breadcrumb,
                 source_url, etc.

    Returns:
        A single formatted string ready to insert into the {context}
        placeholder of the system prompt.
    """
    if not chunks:
        return "(no relevant context was found for this question)"

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        payload = chunk.get("payload", {})

        doc_title  = payload.get("doc_title", "Unknown document")
        breadcrumb = payload.get("breadcrumb", "")
        content    = payload.get("content", "").strip()

        # Guard against an unusually large chunk blowing the context budget
        if len(content) > MAX_CHUNK_CHARS:
            content = content[:MAX_CHUNK_CHARS] + "... [truncated]"

        block = (
            f"[{i}] Source: {doc_title}\n"
            f"    Section: {breadcrumb}\n"
            f"    Content: {content}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


# =============================================================================
# SOURCE METADATA (for the API/UI layer to render citations)
# =============================================================================

def build_source_list(chunks: list[dict]) -> list[dict]:
    """
    Build a clean list of source metadata for the calling layer (api/) to
    render alongside the streamed answer — e.g. as clickable citation
    links in the frontend.

    Kept separate from build_context() because the LLM only needs to SEE
    formatted text, but the API layer needs structured fields (URL,
    title) it can render as UI elements.

    Returns:
        [{"index": 1, "doc_title": ..., "breadcrumb": ..., "source_url": ...}, ...]
    """
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        payload = chunk.get("payload", {})
        sources.append({
            "index":      i,
            "doc_title":  payload.get("doc_title", ""),
            "breadcrumb": payload.get("breadcrumb", ""),
            "source_url": payload.get("source_url", ""),
            "chunk_type": payload.get("chunk_type", ""),
        })
    return sources


# =============================================================================
# GROQ CALL WITH MODEL FALLBACK
# =============================================================================

def _call_groq(client: Groq, model: str, messages: list[dict], stream: bool):
    """
    Make a single Groq chat completion call. Raises on failure — caller
    handles fallback logic.
    """
    return client.chat.completions.create(
        model       = model,
        messages    = messages,
        temperature = TEMPERATURE,
        max_tokens  = MAX_TOKENS,
        stream      = stream,
    )


def generate_with_fallback(messages: list[dict], stream: bool = STREAM_RESPONSE):
    """
    Call Groq with automatic fallback across models.

    Tries GROQ_MODEL first. If that call raises an exception (rate limit,
    model unavailable, transient API error), tries each model in
    GROQ_FALLBACK_MODELS in order before giving up entirely.

    Args:
        messages — list of {"role": ..., "content": ...} dicts (system + user)
        stream   — whether to request a streaming response

    Returns:
        The Groq completion object (streaming iterator if stream=True,
        otherwise a single completion object).

    Raises:
        RuntimeError if every model (primary + all fallbacks) fails.
    """
    client = get_groq_client()
    models_to_try = [GROQ_MODEL] + GROQ_FALLBACK_MODELS

    last_error = None
    for model in models_to_try:
        try:
            response = _call_groq(client, model, messages, stream)
            if model != GROQ_MODEL:
                logger.warning(f"Used fallback model: {model}")
            return response
        except Exception as e:
            logger.warning(f"Model '{model}' failed: {e}")
            last_error = e
            continue

    raise RuntimeError(
        f"All models failed (tried {models_to_try}). Last error: {last_error}"
    )


# =============================================================================
# MAIN ENTRY POINT — used by api/ stage
# =============================================================================

def generate_answer(query: str, chunks: list[dict], stream: bool = STREAM_RESPONSE):
    """
    Generate a grounded answer for a query given retrieved chunks.

    This is the function the api/ stage should import:
        from app.generate.generator import generate_answer
        for token in generate_answer(query, chunks):
            ...

    Args:
        query  — the student's natural language question
        chunks — retrieved chunks from app.retrieve.run.search()
        stream — if True, returns a generator yielding text tokens;
                 if False, returns the full answer as a single string

    Returns:
        If stream=True:  a generator yielding str tokens as they arrive
        If stream=False: a single str containing the complete answer
    """
    system_template = load_prompt()
    context          = build_context(chunks)
    system_prompt    = system_template.format(context=context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    logger.info(f"Generating answer for: '{query[:50]}...' ({len(chunks)} context chunks)")

    response = generate_with_fallback(messages, stream=stream)

    if stream:
        return _stream_tokens(response)
    else:
        return response.choices[0].message.content


def _stream_tokens(response):
    """
    Generator that yields text tokens from a Groq streaming response.
    Separated out so generate_answer() can return either a generator
    (streaming) or a plain string (non-streaming) with a consistent
    calling convention either way.
    """
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# =============================================================================
# CLI — interactive test
# =============================================================================

if __name__ == "__main__":
    from app.generate.config import validate
    from app.retrieve.run import search

    validate()

    query = " ".join(sys.argv[1:]) or "what is the eligibility for foundation level admission"

    print("\n" + "═" * 65)
    print("  Generate Stage — Test")
    print("═" * 65)
    print(f"\n  Query: '{query}'")
    print(f"\n  Retrieving context...")

    chunks = search(query)
    print(f"  Retrieved {len(chunks)} chunks\n")

    print(f"  Sources:")
    for s in build_source_list(chunks):
        print(f"    [{s['index']}] {s['doc_title']} — {s['breadcrumb'][:50]}")

    print(f"\n  Answer:\n  {'─' * 60}")

    for token in generate_answer(query, chunks, stream=True):
        print(token, end="", flush=True)

    print(f"\n  {'─' * 60}")
    print("═" * 65)
