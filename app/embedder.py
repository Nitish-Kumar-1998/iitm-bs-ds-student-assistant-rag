"""
IITM BS RAG Pipeline — Stage 5: Embedder
==========================================
INPUT:  output/chunks/all_chunks.json     (from hyde_generator.py)
OUTPUT: output/chunks/all_chunks_embedded.json

What it does:
  - Embeds every chunk using all-MiniLM-L6-v2 (local, free)
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

Run:
  python embedder.py
"""

import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import (
    ALL_CHUNKS_FILE,
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
# MAIN
# ══════════════════════════════════════════════════════════════════

def run():
    print("\n" + "═" * 65)
    print("  IITM BS RAG Pipeline — Stage 5: Embedder")
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

    # Load model
    print(f"\n  Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"  ✅ Model loaded — dimension: {dim}")

    if dim != EMBEDDING_DIM:
        logger.warning(f"Model dimension {dim} != config EMBEDDING_DIM {EMBEDDING_DIM}")

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
    print(f"\n  Embedding {len(texts)} chunks in batches of {EMBEDDING_BATCH}...")

    embedded_ids  = set(existing_embedded.keys())
    batch_count   = 0

    with tqdm(total=len(needs_embedding), desc="  Embedding") as pbar:
        for i in range(0, len(needs_embedding), EMBEDDING_BATCH):
            batch_chunks = needs_embedding[i:i + EMBEDDING_BATCH]
            batch_texts  = texts[i:i + EMBEDDING_BATCH]

            # Embed batch
            embeddings = model.encode(
                batch_texts,
                batch_size=EMBEDDING_BATCH,
                convert_to_numpy=True,
                normalize_embeddings=True,  # normalize for cosine similarity
                show_progress_bar=False,
            )

            # Attach to chunks
            for j, chunk in enumerate(batch_chunks):
                chunk["embedding"] = embeddings[j].tolist()
                embedded_ids.add(chunk["chunk_id"])

            batch_count += 1
            pbar.update(len(batch_chunks))

            # Save progress every 100 chunks
            if batch_count % (100 // EMBEDDING_BATCH + 1) == 0:
                save_progress(embedded_ids)
                logger.info(f"Progress saved — {len(embedded_ids)} chunks embedded")

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

    # Save
    with open(EMBEDDED_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    size_mb = EMBEDDED_FILE.stat().st_size / (1024 * 1024)

    # Sample verification
    sample = next((c for c in chunks if "embedding" in c), None)

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

    print(f"\n  Saved to: {EMBEDDED_FILE}")
    print(f"\n  Next step: python uploader.py")
    print("═" * 65)


if __name__ == "__main__":
    run()