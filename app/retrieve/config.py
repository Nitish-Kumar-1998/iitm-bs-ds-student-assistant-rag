"""
config.py
---------
Single source of truth for all retrieve stage settings.

No logic lives here — only paths, constants, and environment variables.
Every other file in this package imports from here.

Pipeline stages that use this file:
    retriever.py — Qdrant connection, hybrid search weights, RRF constant
    reranker.py  — cross-encoder model name, rerank top-k
    run.py       — combined retrieval test runner
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent


# =============================================================================
# VECTOR DATABASE — Qdrant
# =============================================================================
# Must match app/ingest/config.py exactly — this stage reads the same
# collection that ingest/uploader.py wrote to.
#
# QDRANT_MODE:
#   "embedded" — local files, no server (development)
#   "server"   — running Qdrant instance via QDRANT_URL (production)

QDRANT_MODE        = os.getenv("QDRANT_MODE", "embedded")
QDRANT_LOCAL_PATH  = str(BASE_DIR.parent / "ingest" / "data" / "qdrant_local")

QDRANT_URL         = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION  = "iitm_bs_docs_chunks"


# =============================================================================
# EMBEDDING — Google Gemini
# =============================================================================
# Queries are embedded with task_type="retrieval_query" (different from
# the "retrieval_document" used during ingest) — Gemini optimises the
# vector differently depending on which side of the search it represents.

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL  = "models/gemini-embedding-001"
EMBEDDING_DIM    = 3072


# =============================================================================
# HYBRID SEARCH — Vector + Sparse (BM25-style)
# =============================================================================
# VECTOR_WEIGHT / BM25_WEIGHT — must match ingest/config.py for consistency,
# though these specific values are only used as a fallback; the primary
# fusion method is RRF (below), which is rank-based and doesn't need weights.

VECTOR_WEIGHT = 0.7
BM25_WEIGHT   = 0.3

# How many candidates to fetch from EACH search method (dense, sparse)
# before fusion. Fetch generously — RRF and reranking need room to work.
RETRIEVAL_TOP_K = 20


# =============================================================================
# RRF FUSION — Reciprocal Rank Fusion
# =============================================================================
# RRF combines two ranked lists (dense results + sparse results) into one
# fused ranking using only rank position, not raw scores — this avoids the
# problem of dense cosine scores and sparse TF scores being on different
# scales and not directly comparable.
#
# Formula per chunk: score = sum( 1 / (k + rank) ) across each list it
# appears in. A chunk ranked #1 in both lists scores highest.
#
# k=60 is the standard constant from the original RRF paper (Cormack et al.)
# and is the industry default — it dampens the influence of very top ranks
# slightly so a chunk doesn't dominate just from being #1 in one list while
# absent from the other.

RRF_K = 60

# Final number of fused candidates passed into the reranker
FUSION_TOP_K = 20


# =============================================================================
# RERANKING — Cross-encoder
# =============================================================================
# Cross-encoders score (query, document) pairs jointly, which is far more
# accurate than comparing independently-computed embeddings (bi-encoder
# search). They are slower, which is why we only rerank the top ~20
# fused candidates rather than the whole collection.
#
# ms-marco-MiniLM-L-6-v2 is a small, fast, well-established reranker
# trained on the MS MARCO passage ranking dataset — strong general-purpose
# choice for English Q&A retrieval, ~80MB, runs fine on CPU.

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Final number of chunks returned to the generate stage after reranking.
RERANK_TOP_K = 5


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
    Check required environment variables and that the Qdrant collection
    is reachable. Called at startup by run.py and each stage's __main__.
    """
    errors = []

    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is not set in .env")

    if QDRANT_MODE == "embedded":
        qdrant_path = Path(QDRANT_LOCAL_PATH)
        if not qdrant_path.exists():
            errors.append(
                f"Embedded Qdrant data not found at {qdrant_path}\n"
                f"     Run the ingest pipeline first: python -m app.ingest.run"
            )
    elif QDRANT_MODE == "server" and not QDRANT_URL:
        errors.append("QDRANT_MODE=server but QDRANT_URL is not set")

    if errors:
        print("\n❌  Retrieve config errors:")
        for e in errors:
            print(f"    → {e}")
        raise SystemExit(1)

    print("\n✅  Retrieve config validated")
    print(f"    Collection:   {QDRANT_COLLECTION}")
    if QDRANT_MODE == "embedded":
        print(f"    Qdrant:       embedded (local files at {QDRANT_LOCAL_PATH})")
    else:
        print(f"    Qdrant:       server ({QDRANT_URL})")
    print(f"    Embedding:    {EMBEDDING_MODEL}  (dim={EMBEDDING_DIM})")
    print(f"    Reranker:     {RERANKER_MODEL}")
    print(f"    RRF k:        {RRF_K}")
    print(f"    Top-k:        retrieve={RETRIEVAL_TOP_K} → fuse={FUSION_TOP_K} → rerank={RERANK_TOP_K}")


if __name__ == "__main__":
    validate()
