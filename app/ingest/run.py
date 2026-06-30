"""
run.py
------
Combined runner for the full ingest pipeline.

Runs all three stages in sequence:
    Stage 1 — chunker.py   : markdown docs → chunks.json
    Stage 2 — embedder.py  : chunks.json   → chunks_embedded.json
    Stage 3 — uploader.py  : chunks_embedded.json → Qdrant

Each stage can also be run independently:
    python -m app.ingest.chunker
    python -m app.ingest.embedder
    python -m app.ingest.uploader

Run the full pipeline:
    python -m app.ingest.run

Flags:
    --skip-chunking    Start from Stage 2 (use existing chunks.json)
    --skip-embedding   Start from Stage 3 (use existing chunks_embedded.json)
"""

import sys
import time
import logging

from app.ingest.config import validate, LOG_LEVEL, LOG_FORMAT

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("ingest.run")


def main() -> None:
    args             = set(sys.argv[1:])
    skip_chunking    = "--skip-chunking"  in args
    skip_embedding   = "--skip-embedding" in args

    # If skipping embedding we must also skip chunking
    if skip_embedding:
        skip_chunking = True

    print("\n" + "█" * 65)
    print("  IITM BS RAG — Ingest Pipeline")
    print("█" * 65)

    if skip_chunking:
        print("  ⏭  Skipping Stage 1 (chunker)")
    if skip_embedding:
        print("  ⏭  Skipping Stage 2 (embedder)")

    # Validate config and environment before touching any API
    validate()

    pipeline_start = time.time()

    # ── Stage 1: Chunker ──────────────────────────────────────────────────
    if not skip_chunking:
        stage_start = time.time()
        from app.ingest.chunker import run as run_chunker
        chunks = run_chunker()
        print(f"\n  ⏱  Stage 1 completed in {time.time() - stage_start:.1f}s")
    else:
        # Load existing chunks so we can report the count
        import json
        from app.ingest.config import CHUNKS_FILE
        if not CHUNKS_FILE.exists():
            print(f"\n  ❌ --skip-chunking set but {CHUNKS_FILE} not found")
            print(f"     Run without --skip-chunking first")
            raise SystemExit(1)
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"\n  ✅ Stage 1 skipped — loaded {len(chunks)} existing chunks")

    # ── Stage 2: Embedder ─────────────────────────────────────────────────
    if not skip_embedding:
        stage_start = time.time()
        from app.ingest.embedder import run as run_embedder
        embedded_chunks = run_embedder()
        elapsed = time.time() - stage_start
        print(f"\n  ⏱  Stage 2 completed in {elapsed:.1f}s")
        print(f"     ({elapsed / max(len(chunks), 1):.2f}s per chunk)")
    else:
        import json
        from app.ingest.config import EMBEDDED_FILE
        if not EMBEDDED_FILE.exists():
            print(f"\n  ❌ --skip-embedding set but {EMBEDDED_FILE} not found")
            print(f"     Run without --skip-embedding first")
            raise SystemExit(1)
        with open(EMBEDDED_FILE, encoding="utf-8") as f:
            embedded_chunks = json.load(f)
        print(f"\n  ✅ Stage 2 skipped — loaded {len(embedded_chunks)} embedded chunks")

    # ── Stage 3: Uploader ─────────────────────────────────────────────────
    stage_start = time.time()
    from app.ingest.uploader import run as run_uploader
    run_uploader()
    print(f"\n  ⏱  Stage 3 completed in {time.time() - stage_start:.1f}s")

    # ── Done ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start

    print("\n" + "█" * 65)
    print(f"  ✅ Ingest pipeline complete")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Chunks:     {len(embedded_chunks)}")
    print(f"\n  Next step: python -m app.retrieve.run")
    print("█" * 65 + "\n")


if __name__ == "__main__":
    main()
