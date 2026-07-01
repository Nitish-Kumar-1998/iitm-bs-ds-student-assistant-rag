"""
run.py
------
Combined runner for the retrieve stage — interactive query testing.

Runs the full retrieval pipeline for a given query:
    1. retriever.retrieve()  — hybrid search (dense + sparse) + RRF fusion
    2. reranker.rerank()     — cross-encoder reranking

This is the single function the generate stage will import and call.

Run as a CLI tool to test retrieval quality before wiring up generation:
    python -m app.retrieve.run "what is the OPPE exam"
    python -m app.retrieve.run                      # uses a default test query
"""

import sys
import time
import logging

from app.retrieve.config import validate, RERANK_TOP_K, LOG_LEVEL, LOG_FORMAT
from app.retrieve.retriever import retrieve
from app.retrieve.reranker import rerank

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("retrieve.run")


# =============================================================================
# MAIN ENTRY POINT — used by generate/ stage
# =============================================================================

def search(query: str, top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    Full retrieval pipeline: hybrid search → RRF fusion → cross-encoder rerank.

    This is the function the generate stage should import:
        from app.retrieve.run import search
        chunks = search("what is the OPPE exam")

    Args:
        query — natural language question
        top_k — number of final chunks to return after reranking

    Returns:
        List of top-k chunks, each a dict with:
            chunk_id, rrf_score, rerank_score, payload
        payload contains content, breadcrumb, source_url, chunk_type, etc.
        — everything the generate stage needs to build a grounded answer.
    """
    candidates = retrieve(query)
    results    = rerank(query, candidates, top_k=top_k)
    return results


# =============================================================================
# CLI — print formatted results for manual quality testing
# =============================================================================

def _print_results(query: str, results: list[dict], elapsed: float) -> None:
    print(f"\n  Query: '{query}'")
    print(f"  Time:  {elapsed:.2f}s")
    print(f"  Top {len(results)} results:\n")

    for i, r in enumerate(results, start=1):
        p = r["payload"]
        print(f"  {'─' * 60}")
        print(
            f"  {i}. [rerank={r['rerank_score']:.3f}  rrf={r.get('rrf_score', 0):.4f}] "
            f"{p.get('chunk_type', '')}"
        )
        print(f"     Doc:        {p.get('doc_title', '')}")
        print(f"     Breadcrumb: {p.get('breadcrumb', '')}")
        print(f"     Source:     {p.get('source_url', '')[:70]}")
        print(f"     Content:    {p.get('content', '')[:200].strip()}...")

    print(f"\n  {'─' * 60}")


def main() -> None:
    validate()

    query = " ".join(sys.argv[1:]) or "what is the eligibility for foundation level admission"

    print("\n" + "═" * 65)
    print("  Retrieve Stage — Hybrid Search + RRF Fusion + Reranking")
    print("═" * 65)

    start = time.time()
    results = search(query)
    elapsed = time.time() - start

    _print_results(query, results, elapsed)
    print("═" * 65)


if __name__ == "__main__":
    main()
