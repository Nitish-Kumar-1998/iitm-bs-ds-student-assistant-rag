"""
retriever.py
------------
Hybrid retrieval: dense (vector) search + sparse (BM25-style) search,
fused with Reciprocal Rank Fusion (RRF).

INPUT:  a natural language query string
OUTPUT: a ranked list of candidate chunks (dicts) with fused RRF scores

Pipeline:
    1. Embed the query via Gemini (task_type="retrieval_query")
    2. Build a sparse TF vector from the query text (same method as ingest)
    3. Search Qdrant on "dense" vector  → ranked list A
    4. Search Qdrant on "sparse" vector → ranked list B
    5. Fuse A and B with RRF            → single ranked list

This module does NOT rerank — that's reranker.py's job. This stage's
output is intentionally wider (FUSION_TOP_K candidates) so the reranker
has enough material to find the true best matches.

Run standalone (interactive query test):
    python -m app.retrieve.retriever "what is the OPPE exam"
"""

import re
import sys
import hashlib
import logging
from collections import Counter

import google.generativeai as genai
from qdrant_client import QdrantClient

from app.retrieve.config import (
    QDRANT_MODE,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
    RRF_K,
    FUSION_TOP_K,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("retrieve.retriever")

_genai_configured = False
_client: QdrantClient | None = None


# =============================================================================
# QDRANT CLIENT (singleton)
# =============================================================================

def get_qdrant_client() -> QdrantClient:
    """
    Return a shared QdrantClient instance, created once and reused.

    Mirrors app/ingest/uploader.py's mode logic exactly — this stage must
    read from the same collection that ingest wrote to, in the same mode
    (embedded local files vs. server).
    """
    global _client
    if _client is not None:
        return _client

    if QDRANT_MODE == "embedded":
        _client = QdrantClient(path=QDRANT_LOCAL_PATH)
        logger.info(f"Connected to embedded Qdrant at {QDRANT_LOCAL_PATH}")
    else:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
        logger.info(f"Connected to Qdrant server at {QDRANT_URL}")

    return _client


# =============================================================================
# QUERY EMBEDDING
# =============================================================================

def embed_query(query: str) -> list[float]:
    """
    Embed a user query via Gemini using task_type="retrieval_query".

    This is intentionally different from task_type="retrieval_document"
    used in ingest/embedder.py — Gemini's embedding model is trained to
    produce asymmetric representations: queries and documents that match
    semantically land close together in vector space, but the model needs
    to know which side of the pair it's embedding to do this well.
    """
    global _genai_configured
    if not _genai_configured:
        genai.configure(api_key=GEMINI_API_KEY)
        _genai_configured = True

    result = genai.embed_content(
        model     = EMBEDDING_MODEL,
        content   = query,
        task_type = "retrieval_query",
    )
    return result["embedding"]


# =============================================================================
# SPARSE QUERY VECTOR
# =============================================================================
# Must use the IDENTICAL hashing scheme as app/ingest/uploader.py's
# build_sparse_vector() — otherwise query terms and document terms would
# land in different hash buckets and sparse search would never match.

def build_sparse_query_vector(text: str) -> tuple[list[int], list[float]]:
    """
    Build a sparse TF vector for the query, using the same hashing scheme
    as ingest/uploader.py so query terms map to the same buckets as the
    document terms they need to match against.
    """
    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    if not words:
        return [0], [0.0]

    tf    = Counter(words)
    total = len(words)

    index_map: dict[int, float] = {}
    for word, count in tf.items():
        idx      = int(hashlib.md5(word.encode()).hexdigest()[:6], 16) % 100_000
        tf_score = count / total
        index_map[idx] = index_map.get(idx, 0.0) + tf_score

    return list(index_map.keys()), [float(v) for v in index_map.values()]


# =============================================================================
# DENSE SEARCH
# =============================================================================

def dense_search(client: QdrantClient, query_vector: list[float], top_k: int) -> list[dict]:
    """
    Vector similarity search on the "dense" embedding field.
    Returns chunks ranked by semantic similarity to the query.
    """
    results = client.query_points(
        collection_name = QDRANT_COLLECTION,
        query            = query_vector,
        using            = "dense",
        limit            = top_k,
        with_payload     = True,
    ).points

    return [
        {"chunk_id": r.payload["chunk_id"], "score": r.score, "payload": r.payload}
        for r in results
    ]


# =============================================================================
# SPARSE SEARCH
# =============================================================================

def sparse_search(client: QdrantClient, sparse_indices: list[int], sparse_values: list[float], top_k: int) -> list[dict]:
    """
    Keyword-style search on the "sparse" TF vector field.
    Returns chunks ranked by term-frequency overlap with the query.
    """
    from qdrant_client.models import SparseVector

    results = client.query_points(
        collection_name = QDRANT_COLLECTION,
        query            = SparseVector(indices=sparse_indices, values=sparse_values),
        using            = "sparse",
        limit            = top_k,
        with_payload     = True,
    ).points

    return [
        {"chunk_id": r.payload["chunk_id"], "score": r.score, "payload": r.payload}
        for r in results
    ]


# =============================================================================
# RRF FUSION
# =============================================================================

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    """
    Fuse multiple ranked lists into one using Reciprocal Rank Fusion.

    Why RRF instead of combining raw scores?
    Dense cosine similarity (0-1 range) and sparse TF scores (unbounded,
    different scale) are not directly comparable — a 0.8 dense score and
    a 0.8 sparse score do not mean the same thing. RRF sidesteps this
    entirely by using only RANK POSITION, which is always comparable.

    Formula: for each chunk, score = sum over all lists it appears in of
        1 / (k + rank_in_that_list)
    Chunks appearing near the top of multiple lists score highest.

    Args:
        ranked_lists — list of ranked result lists (e.g. [dense_results, sparse_results])
        k            — RRF constant, dampens the impact of any single top rank

    Returns:
        Fused list of chunks sorted by RRF score (descending), deduplicated
        by chunk_id, each carrying the full payload from wherever it was
        first seen.
    """
    rrf_scores: dict[str, float] = {}
    chunk_payloads: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in chunk_payloads:
                chunk_payloads[chunk_id] = item["payload"]

    fused = [
        {"chunk_id": cid, "rrf_score": score, "payload": chunk_payloads[cid]}
        for cid, score in rrf_scores.items()
    ]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    return fused


# =============================================================================
# MAIN RETRIEVAL FUNCTION
# =============================================================================

def retrieve(query: str, top_k: int = FUSION_TOP_K) -> list[dict]:
    """
    Run hybrid retrieval for a query: dense + sparse search, fused via RRF.

    This is the primary entry point other modules (reranker.py, generate/)
    should import and call.

    Args:
        query — natural language question
        top_k — number of fused candidates to return (default: FUSION_TOP_K)

    Returns:
        List of dicts: [{"chunk_id", "rrf_score", "payload"}, ...]
        sorted by rrf_score descending. payload contains the full chunk
        (content, breadcrumb, source_url, chunk_type, etc.)
    """
    client = get_qdrant_client()

    # Embed query for dense search
    query_vector = embed_query(query)

    # Build sparse vector for keyword search
    sparse_indices, sparse_values = build_sparse_query_vector(query)

    # Run both searches
    dense_results  = dense_search(client, query_vector, RETRIEVAL_TOP_K)
    sparse_results = sparse_search(client, sparse_indices, sparse_values, RETRIEVAL_TOP_K)

    logger.info(
        f"Query: '{query[:50]}...' → "
        f"dense={len(dense_results)} sparse={len(sparse_results)} candidates"
    )

    # Fuse with RRF
    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=RRF_K)

    return fused[:top_k]


# =============================================================================
# CLI — interactive query test
# =============================================================================

def _print_results(query: str, results: list[dict]) -> None:
    print(f"\n  Query: '{query}'")
    print(f"  Fused candidates: {len(results)}\n")
    for i, r in enumerate(results, start=1):
        p = r["payload"]
        print(
            f"  {i:2d}. [rrf={r['rrf_score']:.4f}] "
            f"{p.get('chunk_type', ''):6s} | {p.get('breadcrumb', '')[:60]}"
        )


if __name__ == "__main__":
    from app.retrieve.config import validate
    validate()

    query = " ".join(sys.argv[1:]) or "what is the eligibility for foundation level"
    results = retrieve(query)
    _print_results(query, results)
