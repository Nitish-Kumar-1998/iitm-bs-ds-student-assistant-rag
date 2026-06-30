"""
embedder.py
-----------
Stage 2 of the ingest pipeline.

Reads chunks.json produced by chunker.py, generates Gemini embeddings
for every chunk, and writes chunks_embedded.json.

INPUT:
    app/ingest/data/chunks.json

OUTPUT:
    app/ingest/data/chunks_embedded.json
    app/ingest/data/embedding_progress.json  (resume support, deleted on success)

Provider: Google Gemini — models/gemini-embedding-001
    - Output dimension: 3072
    - Free tier: 1500 RPD, 100 RPM
    - task_type: retrieval_document (optimised for RAG document embedding)

Resume support:
    If the process is interrupted, re-running embedder.py will skip
    chunks that already have embeddings and continue from where it left off.
    The progress file is deleted automatically on successful completion.

Run standalone:
    python -m app.ingest.embedder

Or as part of the full pipeline:
    python -m app.ingest.run
"""

import json
import time
import logging

import google.generativeai as genai
from tqdm import tqdm

from app.ingest.config import (
    CHUNKS_FILE,
    EMBEDDED_FILE,
    PROGRESS_FILE,
    OUTPUT_DIR,
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_BATCH,
    EMBED_DELAY_SECONDS,
    EMBED_MAX_RETRIES,
    EMBED_RETRY_WAIT,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("ingest.embedder")


# =============================================================================
# EMBED TEXT BUILDER
# =============================================================================

def build_embed_text(chunk: dict) -> str:
    """
    Build the text string that gets sent to the embedding model.

    We use embed_text (heading + breadcrumb + content) as the primary signal.
    This was already assembled by chunker.py and is richer than content alone.

    Why not just embed content?
    Without heading/breadcrumb, two chunks from different sections with
    similar wording would produce nearly identical vectors. The context
    signal (where in the document this came from) is what separates them.
    """
    return chunk.get("embed_text", chunk.get("content", "")).strip()


# =============================================================================
# PROGRESS — RESUME SUPPORT
# =============================================================================

def load_progress() -> dict[str, list[float]]:
    """
    Load previously embedded chunk IDs and their embeddings.

    Returns a dict mapping chunk_id → embedding vector.
    Returns an empty dict if no progress file exists (fresh run).
    """
    if not PROGRESS_FILE.exists():
        return {}

    with open(PROGRESS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    embeddings = data.get("embeddings", {})
    logger.info(f"Resumed — {len(embeddings)} chunks already embedded")
    return embeddings


def save_progress(embeddings: dict[str, list[float]]) -> None:
    """
    Persist progress to disk so a crash does not lose completed embeddings.
    Called every 10 chunks during embedding.
    """
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"embeddings": embeddings}, f)


# =============================================================================
# EMBEDDING WITH RETRY
# =============================================================================

def embed_text_with_retry(text: str) -> list[float]:
    """
    Embed a single text string with exponential-ish retry on failure.

    Why one text at a time?
    EMBEDDING_BATCH = 1 is intentional — Gemini free tier is 100 RPM.
    Batching would hit quota faster and make retry logic more complex.
    One-at-a-time with a short delay stays safely within limits.

    Raises:
        RuntimeError if all retries are exhausted.
    """
    for attempt in range(1, EMBED_MAX_RETRIES + 1):
        try:
            result = genai.embed_content(
                model     = EMBEDDING_MODEL,
                content   = text,
                task_type = "retrieval_document",
            )
            return result["embedding"]

        except Exception as e:
            if attempt == EMBED_MAX_RETRIES:
                raise RuntimeError(
                    f"Embedding failed after {EMBED_MAX_RETRIES} retries: {e}"
                ) from e
            logger.warning(
                f"Embed attempt {attempt}/{EMBED_MAX_RETRIES} failed: {e} "
                f"— retrying in {EMBED_RETRY_WAIT}s"
            )
            time.sleep(EMBED_RETRY_WAIT)

    # Unreachable but satisfies type checkers
    raise RuntimeError("Embedding failed")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_embeddings(chunks: list[dict]) -> int:
    """
    Check that every chunk has a valid embedding of the expected dimension.

    Returns:
        Number of validation issues found (0 = all good).
    """
    issues = 0
    for chunk in chunks:
        embedding = chunk.get("embedding")
        if not embedding:
            logger.warning(f"Missing embedding: chunk_id={chunk.get('chunk_id', '?')}")
            issues += 1
            continue
        if len(embedding) != EMBEDDING_DIM:
            logger.warning(
                f"Dimension mismatch: chunk_id={chunk.get('chunk_id', '?')} "
                f"got {len(embedding)}, expected {EMBEDDING_DIM}"
            )
            issues += 1
    return issues


# =============================================================================
# MAIN
# =============================================================================

def run() -> list[dict]:
    """
    Entry point for Stage 2: Embedder.

    Loads chunks from CHUNKS_FILE, embeds each one via Gemini,
    attaches the embedding vector to the chunk dict, and saves to EMBEDDED_FILE.

    Supports resuming from a previous interrupted run via PROGRESS_FILE.

    Returns the final list of embedded chunks (used by run.py to chain stages).
    """
    print("\n" + "═" * 65)
    print("  Stage 2 — Embedder")
    print(f"  Model:      {EMBEDDING_MODEL}  (dim={EMBEDDING_DIM})")
    print(f"  Reading from: {CHUNKS_FILE}")
    print(f"  Writing to:   {EMBEDDED_FILE}")
    print("═" * 65)

    # ── Load chunks ───────────────────────────────────────────────────────
    if not CHUNKS_FILE.exists():
        print(f"\n  ❌ {CHUNKS_FILE} not found")
        print(f"     Run chunker first: python -m app.ingest.chunker")
        raise SystemExit(1)

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"\n  Loaded {len(chunks)} chunks")

    # ── Resume support ────────────────────────────────────────────────────
    cached_embeddings = load_progress()

    # Also check existing embedded file for any completed embeddings
    if EMBEDDED_FILE.exists() and not cached_embeddings:
        print(f"  Found existing embedded file — loading for resume...")
        with open(EMBEDDED_FILE, encoding="utf-8") as f:
            existing = json.load(f)
        cached_embeddings = {
            c["chunk_id"]: c["embedding"]
            for c in existing
            if "embedding" in c
        }
        print(f"  Already embedded: {len(cached_embeddings)} chunks")

    # Separate chunks into already-done and needs-embedding
    needs_embedding = []
    already_done    = 0

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id in cached_embeddings:
            chunk["embedding"] = cached_embeddings[chunk_id]
            already_done += 1
        else:
            needs_embedding.append(chunk)

    print(f"  Already embedded: {already_done}")
    print(f"  To embed:         {len(needs_embedding)}")

    if not needs_embedding:
        print(f"\n  ✅ All chunks already embedded — nothing to do")
        _save_and_report(chunks)
        return chunks

    # ── Initialise Gemini ─────────────────────────────────────────────────
    print(f"\n  Initialising Gemini...")
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"  ✅ Gemini ready")

    # ── Embed ─────────────────────────────────────────────────────────────
    print(f"\n  Embedding {len(needs_embedding)} chunks...\n")

    save_every = 10   # save progress every N chunks

    with tqdm(total=len(needs_embedding), desc="  Embedding", unit="chunk") as pbar:
        for i, chunk in enumerate(needs_embedding):
            text = build_embed_text(chunk)

            embedding = embed_text_with_retry(text)
            chunk["embedding"] = embedding
            cached_embeddings[chunk["chunk_id"]] = embedding

            pbar.update(1)

            # Save progress periodically so a crash loses at most save_every chunks
            if (i + 1) % save_every == 0:
                save_progress(cached_embeddings)

            # Gentle rate limiting — stay well within 100 RPM free tier
            time.sleep(EMBED_DELAY_SECONDS)

    # Final progress save before writing output
    save_progress(cached_embeddings)

    # ── Save and report ───────────────────────────────────────────────────
    _save_and_report(chunks)

    # Clean up progress file — only on full success
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        logger.info("Progress file cleaned up")

    return chunks


def _save_and_report(chunks: list[dict]) -> None:
    """
    Validate, save, and print a summary of the embedded chunks.
    Extracted so both the full-run and already-done paths share the same output.
    """
    # Validate dimensions
    issues = validate_embeddings(chunks)

    # Check actual dimension against config
    sample = next((c for c in chunks if "embedding" in c), None)
    if sample:
        actual_dim = len(sample["embedding"])
        if actual_dim != EMBEDDING_DIM:
            logger.warning(
                f"Actual embedding dim ({actual_dim}) differs from "
                f"config EMBEDDING_DIM ({EMBEDDING_DIM}). "
                f"Update config.py: EMBEDDING_DIM = {actual_dim}"
            )

    # Save
    with open(EMBEDDED_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    size_mb = EMBEDDED_FILE.stat().st_size / (1024 * 1024)

    print(f"\n  {'─' * 40}")
    print(f"  Total chunks:      {len(chunks)}")
    print(f"  With embeddings:   {sum(1 for c in chunks if 'embedding' in c)}")
    print(f"  Missing:           {sum(1 for c in chunks if 'embedding' not in c)}")
    print(f"  Validation issues: {issues}")
    print(f"  File size:         {size_mb:.1f} MB")

    if sample:
        print(f"\n  Sample:")
        print(f"    chunk_id:   {sample['chunk_id']}")
        print(f"    chunk_type: {sample['chunk_type']}")
        print(f"    embed_dim:  {len(sample['embedding'])}")
        print(f"    embed_text: {sample.get('embed_text', '')[:60]}...")

    print(f"\n  Saved → {EMBEDDED_FILE}")
    print("═" * 65)


if __name__ == "__main__":
    from app.ingest.config import validate
    validate()
    run()
