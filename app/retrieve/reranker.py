"""
reranker.py
-----------
Cross-encoder reranking of fused retrieval candidates.

INPUT:  query string + list of fused candidate chunks (from retriever.py)
OUTPUT: top-N chunks reordered by true relevance to the query

Why rerank at all?
Dense and sparse search both compute query and document representations
INDEPENDENTLY, then compare them (bi-encoder style) — fast, but limited,
because the model never actually looks at the query and document together.

A cross-encoder takes (query, document) as a single joint input and
outputs a direct relevance score. This is far more accurate but much
slower — too slow to run over an entire collection, which is why we only
rerank the ~20 candidates that already survived hybrid retrieval + RRF.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    Trained on MS MARCO passage ranking — strong general-purpose choice
    for English Q&A relevance scoring. ~80MB, runs fine on CPU, no GPU
    required for this dataset size.

Run standalone (interactive query + rerank test):
    python -m app.retrieve.reranker "what is the OPPE exam"
"""

import sys
import logging

from sentence_transformers import CrossEncoder

from app.retrieve.config import (
    RERANKER_MODEL,
    RERANK_TOP_K,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("retrieve.reranker")

# Module-level singleton — the cross-encoder model is loaded once and reused
# across calls. Loading it fresh per query would add several seconds of
# overhead every single time.
_model: CrossEncoder | None = None


# =============================================================================
# MODEL LOADER (singleton)
# =============================================================================

def get_reranker_model() -> CrossEncoder:
    """
    Load and cache the cross-encoder model.

    First call downloads ~80MB from HuggingFace (cached locally afterward
    under ~/.cache/torch/sentence_transformers/). Subsequent calls reuse
    the in-memory model — no reload cost.
    """
    global _model
    if _model is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL} (first run downloads ~80MB)")
        _model = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker model loaded")
    return _model


# =============================================================================
# RERANK
# =============================================================================

def rerank(query: str, candidates: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    Rerank fused candidates by true (query, document) relevance.

    Args:
        query      — the original natural language question
        candidates — list of dicts from retriever.retrieve(), each with
                     a "payload" containing "content" (and "embed_text")
        top_k      — number of top-reranked chunks to return

    Returns:
        Top-k candidates from the input list, reordered by rerank_score
        (descending), each dict gaining a new "rerank_score" key.

    Content used for scoring:
        We score against payload["embed_text"] rather than raw "content",
        since embed_text already includes heading + breadcrumb context —
        the same richer signal used during embedding. This helps the
        cross-encoder understand WHERE a chunk sits in the document, not
        just what it says.
    """
    if not candidates:
        logger.warning("No candidates to rerank")
        return []

    model = get_reranker_model()

    # Build (query, document) pairs for the cross-encoder
    pairs = [
        (query, c["payload"].get("embed_text") or c["payload"].get("content", ""))
        for c in candidates
    ]

    scores = model.predict(pairs)

    # Attach scores and sort
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    logger.info(
        f"Reranked {len(candidates)} candidates → "
        f"top score={reranked[0]['rerank_score']:.3f}, "
        f"returning top {min(top_k, len(reranked))}"
    )

    return reranked[:top_k]


# =============================================================================
# CLI — interactive retrieve + rerank test
# =============================================================================

def _print_results(query: str, results: list[dict]) -> None:
    print(f"\n  Query: '{query}'")
    print(f"  Reranked top {len(results)}:\n")
    for i, r in enumerate(results, start=1):
        p = r["payload"]
        print(
            f"  {i:2d}. [rerank={r['rerank_score']:.3f}  rrf={r.get('rrf_score', 0):.4f}] "
            f"{p.get('chunk_type', ''):6s} | {p.get('breadcrumb', '')[:55]}"
        )
        print(f"      {p.get('content', '')[:120].strip()}...")


if __name__ == "__main__":
    from app.retrieve.config import validate
    from app.retrieve.retriever import retrieve

    validate()

    query = " ".join(sys.argv[1:]) or "what is the eligibility for foundation level"

    candidates = retrieve(query)
    results = rerank(query, candidates)

    _print_results(query, results)
