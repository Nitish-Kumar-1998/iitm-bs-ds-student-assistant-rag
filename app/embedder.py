"""
IITM BS RAG Pipeline — Stage 5: Embedder
==========================================
INPUT:  output/chunks/all_chunks.json     (from hyde_generator.py)
OUTPUT: output/chunks/all_chunks_embedded.json

What it does:
  - Embeds every chunk using Voyage AI voyage-3 (API, free tier)
  - Embedding text = embed_text + hyde_questions combined
    (richer signal — content + questions that point to it)
  - Progress saved every 100 chunks — crash safe, resume on restart
  - Skips already-embedded chunks on resume

Why embed hyde_questions too:
  At retrieval time, student question is compared against:
  1. embed_text  (heading + breadcrumb + content)
  2. hyde_questions (pre-generated questions about this chunk)
  Combined embedding captures both what the chunk SAYS
  and what questions it ANSWERS — much better retrieval.

Why Voyage AI:
  - voyage-3 is 21% better than MiniLM on MTEB benchmarks
  - Zero RAM overhead — models run on Voyage AI servers
  - input_type="document" for upload, input_type="query" for search
    (asymmetric embeddings improve retrieval quality)
  - 50M tokens free per month

Run:
  python embedder.py
"""

import json
import logging
import time
import voyageai
from pathlib import Path
from tqdm import tqdm

from config import (
    ALL_CHUNKS_FILE,
    VOYAGE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_BATCH,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("embedder")

# ══════════════════════════════════════════════════════════════════
# OUTPUT FILE
# Separate from all_chunks.json — keeps embeddings out of
# the human-readable chunks file
# ══════════════════════════════════════════════════════════════════

EMBEDDED_FILE = ALL_CHUNKS_FILE.parent / "all_chunks_embedded.json"
PROGRESS_FILE = ALL_CHUNKS_FILE.parent / "embedding_progress.json"

# Voyage AI batch limit — max 128 texts per request
VOYAGE_BATCH_SIZE = 128


# ══════════════════════════════════════════════════════════════════
# EMBED TEXT BUILDER
# Combines embed_text + hyde_questions for richer signal
# ══════════════════════════════════════════════════════════════════

def build_embed_text(chunk: dict) -> str:
    """
    INPUT:  chunk dict
    OUTPUT: combined text to embed

    Combines:
      1. embed_text (heading + breadcrumb + content)
      2. hyde_questions (pre-generated questions)

    This means the embedding captures:
      - What the chunk contains
      - What questions it answers
    → Student questions match on both dimensions
    """
    parts = []

    embed_text = chunk.get("embed_text", "").strip()
    if embed_text:
        parts.append(embed_text)

    hyde_questions = chunk.get("hyde_questions", [])
    if hyde_questions:
        parts.append(" ".join(hyde_questions))

    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════
# PROGRESS — crash safe
# ══════════════════════════════════════════════════════════════════

def load_progress() -> set:
    """Load set of already-embedded chunk_ids."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        logger.info(f"Resuming — {len(data['embedded'])} chunks already done")
        return set(data["embedded"])
    return set()


def save_progress(embedded_ids: set):
    """Save set of embedded chunk_ids."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"embedded": list(embedded_ids)}, f)


# ══════════════════════════════════════════════════════════════════
# VOYAGE AI EMBED WITH RETRY
# ══════════════════════════════════════════════════════════════════

def embed_batch_with_retry(
    client: voyageai.Client,
    texts: list[str],
    model: str,
    input_type: str = "document",
    max_retries: int = 3,
) -> list[list[float]]:
    """
    Embed a batch of texts with Voyage AI.
    Retries on rate limit or transient errors.

    input_type:
      "document" → for uploading chunks to Qdrant (richer representation)
      "query"    → for search time queries (optimised for retrieval)
    """
    for attempt in range(max_retries):
        try:
            result = client.embed(
                texts,
                model=model,
                input_type=input_type,
            )
            return result.embeddings
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err:
                wait = 2 ** attempt
                logger.warning(f"Rate limited — waiting {wait}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait)
            else:
                logger.error(f"Embedding failed: {e}")
                raise e
    raise Exception(f"Embedding failed after {max_retries} retries")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run():
    print("\n" + "═" * 65)
    print("  IITM BS RAG Pipeline — Stage 5: Embedder")
    print("  Provider: Voyage AI (voyage-3)")
    print("═" * 65)

    # Load chunks
    if not ALL_CHUNKS_FILE.exists():
        print(f"\n  ❌ {ALL_CHUNKS_FILE} not found")
        print(f"     Run python hyde_generator.py first")
        return

    with open(ALL_CHUNKS_FILE) as f:
        chunks = json.load(f)
    print(f"\n  Loaded {len(chunks)} chunks")

    # Load existing embeddings if resuming
    existing_embedded = {}
    if EMBEDDED_FILE.exists():
        print(f"  Found existing embedded file — loading for resume...")
        with open(EMBEDDED_FILE) as f:
            existing = json.load(f)
        existing_embedded = {c["chunk_id"]: c["embedding"] for c in existing if "embedding" in c}
        print(f"  Already embedded: {len(existing_embedded)} chunks")

    # Initialise Voyage AI client
    print(f"\n  Initialising Voyage AI client...")
    client = voyageai.Client(api_key=VOYAGE_API_KEY)
    print(f"  ✅ Voyage AI ready — model: {EMBEDDING_MODEL} (dim: {EMBEDDING_DIM})")

    # Split into needs embedding vs already done
    needs_embedding = []
    already_done    = 0

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id in existing_embedded:
            chunk["embedding"] = existing_embedded[chunk_id]
            already_done += 1
        else:
            needs_embedding.append(chunk)

    print(f"  Already embedded: {already_done}")
    print(f"  To embed:         {len(needs_embedding)}")

    if not needs_embedding:
        print(f"\n  ✅ All chunks already embedded")
        _save_and_report(chunks)
        return

    # Build texts to embed
    texts = [build_embed_text(c) for c in needs_embedding]

    # Stats on hyde_questions coverage
    with_hyde    = sum(1 for c in needs_embedding if c.get("hyde_questions"))
    without_hyde = sum(1 for c in needs_embedding if not c.get("hyde_questions"))
    print(f"\n  HyDE coverage:")
    print(f"    With questions:    {with_hyde}")
    print(f"    Without questions: {without_hyde} (embed_text only)")

    # Embed in batches with progress saving
    print(f"\n  Embedding {len(texts)} chunks via Voyage AI")
    print(f"  Batch size: {VOYAGE_BATCH_SIZE} | input_type: document")
    print(f"  (input_type='document' = optimised for storage/retrieval)\n")

    embedded_ids = set(existing_embedded.keys())
    batch_count  = 0

    with tqdm(total=len(needs_embedding), desc="  Embedding") as pbar:
        for i in range(0, len(needs_embedding), VOYAGE_BATCH_SIZE):
            batch_chunks = needs_embedding[i:i + VOYAGE_BATCH_SIZE]
            batch_texts  = texts[i:i + VOYAGE_BATCH_SIZE]

            # Embed batch via Voyage AI
            # input_type="document" → optimised for document storage
            # at search time, query uses input_type="query" → asymmetric retrieval
            embeddings = embed_batch_with_retry(
                client=client,
                texts=batch_texts,
                model=EMBEDDING_MODEL,
                input_type="document",
            )

            # Attach to chunks
            for j, chunk in enumerate(batch_chunks):
                chunk["embedding"] = embeddings[j]
                embedded_ids.add(chunk["chunk_id"])

            batch_count += 1
            pbar.update(len(batch_chunks))

            # Save progress every 5 batches (~640 chunks)
            if batch_count % 5 == 0:
                save_progress(embedded_ids)
                logger.info(f"Progress saved — {len(embedded_ids)} chunks embedded")

            # Small delay to respect rate limits
            time.sleep(0.1)

    # Final save
    save_progress(embedded_ids)
    _save_and_report(chunks)

    # Clean up progress file on success
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        logger.info("Progress file cleaned up")


def _save_and_report(chunks: list[dict]):
    """Save embedded chunks and print summary."""

    # Validate all chunks have embeddings
    missing = [c["chunk_id"] for c in chunks if "embedding" not in c]
    if missing:
        logger.warning(f"{len(missing)} chunks missing embeddings!")

    # Validate embedding dimension matches config
    sample = next((c for c in chunks if "embedding" in c), None)
    if sample:
        actual_dim = len(sample["embedding"])
        if actual_dim != EMBEDDING_DIM:
            logger.warning(
                f"Embedding dim mismatch: got {actual_dim}, config says {EMBEDDING_DIM}. "
                f"Update EMBEDDING_DIM in config.py to {actual_dim}"
            )

    # Save
    with open(EMBEDDED_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    size_mb = EMBEDDED_FILE.stat().st_size / (1024 * 1024)

    print(f"\n  {'═' * 40}")
    print(f"  Total chunks:      {len(chunks)}")
    print(f"  With embeddings:   {sum(1 for c in chunks if 'embedding' in c)}")
    print(f"  Missing:           {len(missing)}")
    print(f"  File size:         {size_mb:.1f} MB")

    if sample:
        print(f"\n  Sample verification:")
        print(f"    chunk_id:    {sample['chunk_id']}")
        print(f"    chunk_type:  {sample['chunk_type']}")
        print(f"    embed_dim:   {len(sample['embedding'])}")
        print(f"    hyde_q:      {len(sample.get('hyde_questions', []))} questions")
        print(f"    embed_text:  {sample.get('embed_text','')[:60]}...")
        print(f"    provider:    Voyage AI ({EMBEDDING_MODEL})")

    print(f"\n  Saved to: {EMBEDDED_FILE}")
    print(f"\n  Next step: python uploader.py")
    print("═" * 65)


if __name__ == "__main__":
    run()    