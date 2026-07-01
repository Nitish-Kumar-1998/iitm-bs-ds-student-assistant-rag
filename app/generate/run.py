"""
run.py
------
Combined runner for the generate stage — full RAG pipeline test.

Chains all three stages together:
    1. app.retrieve.run.search()       — hybrid search + RRF + rerank
    2. app.generate.generator.generate_answer() — grounded streamed answer

This is the single function the api/ stage will import and call to go
from a raw question to a complete answer.

Run as a CLI tool to test the full pipeline end-to-end:
    python -m app.generate.run "what is the OPPE exam"
    python -m app.generate.run                          # default test query
"""

import sys
import time
import logging

from app.generate.config import validate, CONTEXT_CHUNK_COUNT, LOG_LEVEL, LOG_FORMAT
from app.generate.generator import generate_answer, build_source_list
from app.retrieve.run import search

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("generate.run")


# =============================================================================
# MAIN ENTRY POINT — used by api/ stage
# =============================================================================

def ask(query: str, stream: bool = True):
    """
    Full RAG pipeline: retrieve relevant chunks, then generate a grounded
    answer from them.

    This is the function the api/ stage should import:
        from app.generate.run import ask
        answer_stream, sources = ask("what is the OPPE exam")

    Args:
        query  — the student's natural language question
        stream — if True, returns a token generator; if False, a full string

    Returns:
        (answer, sources) tuple where:
            answer  — generator of str tokens (if stream=True) or a full
                      str answer (if stream=False)
            sources — list of source metadata dicts (see
                      generator.build_source_list) for citation rendering
    """
    chunks  = search(query, top_k=CONTEXT_CHUNK_COUNT)
    sources = build_source_list(chunks)
    answer  = generate_answer(query, chunks, stream=stream)

    return answer, sources


# =============================================================================
# CLI — full pipeline test
# =============================================================================

def main() -> None:
    validate()

    query = " ".join(sys.argv[1:]) or "what is the eligibility for foundation level admission"

    print("\n" + "█" * 65)
    print("  Full RAG Pipeline — Retrieve + Generate")
    print("█" * 65)
    print(f"\n  Query: '{query}'\n")

    start = time.time()
    answer_stream, sources = ask(query, stream=True)

    print(f"  Sources retrieved:")
    for s in sources:
        print(f"    [{s['index']}] {s['doc_title']} — {s['breadcrumb'][:55]}")

    print(f"\n  Answer:")
    print(f"  {'─' * 60}")

    full_answer = ""
    for token in answer_stream:
        print(token, end="", flush=True)
        full_answer += token

    elapsed = time.time() - start

    print(f"\n  {'─' * 60}")
    print(f"\n  Total time: {elapsed:.2f}s")
    print("█" * 65 + "\n")


if __name__ == "__main__":
    main()
