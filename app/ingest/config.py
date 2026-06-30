"""
config.py
---------
Single source of truth for all ingest pipeline settings.

No logic lives here — only paths, constants, and environment variables.
Every other file in this package imports from here.
To change any setting, change it here. Nothing else needs to be touched.

Pipeline stages that use this file:
    chunker.py   — DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_TOKENS
    embedder.py  — GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH
    uploader.py  — QDRANT_*, EMBEDDING_DIM, VECTOR_WEIGHT, BM25_WEIGHT
    run.py       — all of the above (combined runner)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root (two levels up from this file)
# app/ingest/config.py → app/ → project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


# =============================================================================
# PATHS
# =============================================================================
# BASE_DIR       — this file's directory (app/ingest/)
# SCRAPER_DIR    — the scraper package (app/scraper/)
# SCRAPER_DATA   — all scraper outputs live here (app/scraper/data/)
# DOCS_DIR       — markdown files produced by the scraper
# REGISTRY_FILE  — registry.json: index of every scraped document
# OUTPUT_DIR     — intermediate ingest outputs (gitignored)
# CHUNKS_FILE    — stage 1 output: raw chunks before embedding
# EMBEDDED_FILE  — stage 2 output: chunks with embeddings attached

BASE_DIR        = Path(__file__).parent
SCRAPER_DIR     = BASE_DIR.parent / "scraper"
SCRAPER_DATA    = SCRAPER_DIR / "data"

DOCS_DIR        = SCRAPER_DATA / "docs"          # contains root/ and nested/
REGISTRY_FILE   = SCRAPER_DATA / "registry.json"

OUTPUT_DIR      = BASE_DIR / "data"
CHUNKS_FILE     = OUTPUT_DIR / "chunks.json"
EMBEDDED_FILE   = OUTPUT_DIR / "chunks_embedded.json"
PROGRESS_FILE   = OUTPUT_DIR / "embedding_progress.json"  # resume support


# =============================================================================
# CHUNKING
# =============================================================================
# CHUNK_SIZE      — max tokens per chunk (1 token ≈ 4 chars)
#                   800 chosen because IITM docs contain rich tables and
#                   long FAQ sections; 512 was too small to fit a single table.
#
# CHUNK_OVERLAP   — tokens of overlap between adjacent text chunks.
#                   100 ensures a sentence at a chunk boundary is not lost
#                   from both sides. Standard production value for doc RAG.
#
# MIN_CHUNK_TOKENS — chunks smaller than this are discarded as noise
#                    (e.g. empty sections, lone headings with no content).

CHUNK_SIZE          = 800
CHUNK_OVERLAP       = 100
MIN_CHUNK_TOKENS    = 30


# =============================================================================
# EMBEDDING — Google Gemini
# =============================================================================
# Free tier: 1500 RPD, 100 RPM — sufficient for this dataset size.
# EMBEDDING_BATCH = 1 is intentionally conservative to stay well within
# rate limits and allow clean retry logic per chunk.
#
# EMBED_DELAY_SECONDS — pause between API calls to avoid 429 errors.

GEMINI_API_KEY          = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL         = "models/gemini-embedding-001"
EMBEDDING_DIM           = 3072   # fixed output dimension for this model
EMBEDDING_BATCH         = 1      # one chunk per API call (rate limit safety)
EMBED_DELAY_SECONDS     = 0.5    # seconds between calls
EMBED_MAX_RETRIES       = 5      # retry attempts before raising
EMBED_RETRY_WAIT        = 10     # seconds to wait between retries


# =============================================================================
# VECTOR DATABASE — Qdrant
# =============================================================================
# Collection name is explicit about its contents:
#   iitm_bs_docs_chunks → IITM BS programme docs, chunked for RAG
#
# VECTOR_WEIGHT / BM25_WEIGHT — hybrid search fusion weights.
#   0.7 / 0.3 is the standard starting point; vector search is primary
#   because semantic understanding matters more than keyword overlap here.
#
# RETRIEVAL_TOP_K — candidates fetched from Qdrant before reranking.
#                   Fetch more than needed so reranker has room to work.
# RERANK_TOP_K    — final chunks passed to the LLM after reranking.
#
# QDRANT_MODE:
#   "embedded" — no server needed. qdrant-client stores data as local files
#                under QDRANT_LOCAL_PATH. Used for development before a
#                server (Docker / cloud) is set up.
#   "server"   — connects to a running Qdrant instance via QDRANT_URL
#                (local Docker or Qdrant Cloud). Used in production.
#
# Switch modes by setting QDRANT_MODE in .env. Defaults to "embedded"
# so the project works out of the box with zero infrastructure.

QDRANT_MODE         = os.getenv("QDRANT_MODE", "embedded")  # "embedded" | "server"
QDRANT_LOCAL_PATH   = str(BASE_DIR / "data" / "qdrant_local")

QDRANT_URL          = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY      = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION   = "iitm_bs_docs_chunks"

VECTOR_WEIGHT       = 0.7
BM25_WEIGHT         = 0.3

RETRIEVAL_TOP_K     = 20   # candidates before reranking
RERANK_TOP_K        = 5    # final chunks passed to LLM

UPLOAD_BATCH_SIZE   = 100  # points per Qdrant upsert call


# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL   = "INFO"
LOG_FORMAT  = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


# =============================================================================
# VALIDATION
# =============================================================================

def validate() -> None:
    """
    Check that all required environment variables are set and critical
    paths exist. Called at startup by run.py and each stage's __main__.

    Raises SystemExit(1) on any error so the pipeline fails fast
    rather than hitting an API error mid-run.
    """
    errors   = []
    warnings = []

    # Required secrets
    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is not set in .env")

    # Qdrant: only require URL when running in server mode
    if QDRANT_MODE == "server" and not QDRANT_URL:
        errors.append("QDRANT_MODE=server but QDRANT_URL is not set")

    # Scraper output must exist before ingest can run
    if not DOCS_DIR.exists():
        errors.append(
            f"Scraper output not found: {DOCS_DIR}\n"
            f"     Run the scraper first: python -m app.scraper.scraper"
        )

    if not REGISTRY_FILE.exists():
        warnings.append(
            f"registry.json not found at {REGISTRY_FILE} — "
            f"doc metadata will fall back to frontmatter only"
        )

    if errors:
        print("\n❌  Ingest config errors:")
        for e in errors:
            print(f"    → {e}")
        raise SystemExit(1)

    if warnings:
        print("\n⚠️   Ingest config warnings:")
        for w in warnings:
            print(f"    → {w}")

    print("\n✅  Ingest config validated")
    print(f"    Docs dir:     {DOCS_DIR}")
    print(f"    Output dir:   {OUTPUT_DIR}")
    print(f"    Collection:   {QDRANT_COLLECTION}")
    print(f"    Embedding:    {EMBEDDING_MODEL}  (dim={EMBEDDING_DIM})")
    if QDRANT_MODE == "embedded":
        print(f"    Qdrant:       embedded (local files at {QDRANT_LOCAL_PATH})")
    else:
        print(f"    Qdrant:       server ({QDRANT_URL})")
    print(f"    Chunk size:   {CHUNK_SIZE} tokens  (overlap={CHUNK_OVERLAP})")


if __name__ == "__main__":
    validate()