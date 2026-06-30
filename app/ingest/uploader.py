"""
uploader.py
-----------
Stage 3 of the ingest pipeline.

Reads chunks_embedded.json, builds Qdrant points (dense + sparse vectors),
deletes the old collection, recreates it, and uploads all points.

INPUT:
    app/ingest/data/chunks_embedded.json

OUTPUT:
    Qdrant collection: iitm_bs_docs_chunks

Vector strategy — Hybrid (dense + sparse):
    dense  — Gemini embedding (3072d cosine) — semantic similarity
    sparse — TF-based sparse vector          — keyword / BM25-style matching

    At query time, both are searched and scores are fused with RRF
    (Reciprocal Rank Fusion) in the retrieve stage.

Payload indexes created for fast filtered search:
    chunk_type, doc_title, root_doc_title, depth, breadcrumb (keyword)
    heading_level, token_count (integer)

Run standalone:
    python -m app.ingest.uploader

Or as part of the full pipeline:
    python -m app.ingest.run
"""

import re
import json
import hashlib
import logging
from collections import Counter

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    PayloadSchemaType,
)

from app.ingest.config import (
    EMBEDDED_FILE,
    QDRANT_MODE,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_DIM,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    UPLOAD_BATCH_SIZE,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("ingest.uploader")


# =============================================================================
# QDRANT CLIENT
# =============================================================================

def get_qdrant_client() -> QdrantClient:
    """
    Build a QdrantClient based on QDRANT_MODE.

    "embedded" — no server required. Data is stored as local files under
                 QDRANT_LOCAL_PATH (app/ingest/data/qdrant_local/).
                 Perfect for development before any infrastructure is set up.

    "server"   — connects to a running Qdrant instance (local Docker or
                 Qdrant Cloud) via QDRANT_URL + QDRANT_API_KEY.
                 Used once you're ready to deploy or share data across runs.

    Switch modes via QDRANT_MODE in .env — no code changes needed.
    """
    if QDRANT_MODE == "embedded":
        from pathlib import Path
        Path(QDRANT_LOCAL_PATH).mkdir(parents=True, exist_ok=True)
        print(f"  Mode: embedded (local files at {QDRANT_LOCAL_PATH})")
        return QdrantClient(path=QDRANT_LOCAL_PATH)

    print(f"  Mode: server ({QDRANT_URL})")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)


# =============================================================================
# POINT ID
# =============================================================================

def chunk_id_to_point_id(chunk_id: str) -> int:
    """
    Convert a 12-char hex chunk_id to a Qdrant point ID (uint64).

    Qdrant requires integer point IDs. We derive them from the chunk_id
    (which is itself an MD5 hash) so IDs are stable across re-ingests —
    the same chunk always gets the same point ID, enabling upsert semantics.

    We take the first 8 hex chars (32 bits) to stay within safe int range.
    Collision probability across ~10k chunks is negligible (~1 in 4 billion).
    """
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:8], 16)


# =============================================================================
# SPARSE VECTOR
# =============================================================================

def build_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """
    Build a sparse TF vector from text for BM25-style keyword matching.

    Why sparse vectors?
    Dense vectors (Gemini embeddings) capture semantic meaning but can miss
    exact keyword matches — e.g. a student searching "OPPE" might not get
    results if the embedding space doesn't cluster that acronym well.
    Sparse vectors ensure exact keyword hits are always retrieved.

    Implementation:
        1. Tokenise to lowercase words (2+ chars, alpha only)
        2. Compute term frequency (TF) per word
        3. Map each word to a stable integer index via MD5 hash % 100000
        4. Collisions are summed (rare, acceptable for this corpus size)

    Returns:
        (indices, values) — Qdrant sparse vector format
    """
    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    if not words:
        # Qdrant requires at least one entry in a sparse vector
        return [0], [0.0]

    tf    = Counter(words)
    total = len(words)

    index_map: dict[int, float] = {}
    for word, count in tf.items():
        # Stable index — same word always maps to the same bucket
        idx      = int(hashlib.md5(word.encode()).hexdigest()[:6], 16) % 100_000
        tf_score = count / total
        index_map[idx] = index_map.get(idx, 0.0) + tf_score

    return list(index_map.keys()), [float(v) for v in index_map.values()]


# =============================================================================
# PAYLOAD BUILDER
# =============================================================================

def build_payload(chunk: dict) -> dict:
    """
    Build the Qdrant point payload from a chunk dict.

    The payload is what gets returned at query time alongside the score.
    We store all metadata needed by the retrieve and generate stages
    so they never need to go back to disk.

    chunk_type-specific fields are included only where relevant to keep
    payload sizes reasonable.
    """
    chunk_type = chunk.get("chunk_type", "text")

    payload = {
        # Identity
        "chunk_id":           chunk.get("chunk_id", ""),
        "chunk_type":         chunk_type,
        "version":            chunk.get("version", "2"),

        # Content
        "content":            chunk.get("content", ""),
        "embed_text":         chunk.get("embed_text", ""),

        # Document location
        "heading":            chunk.get("heading", ""),
        "heading_level":      chunk.get("heading_level", 0),
        "section":            chunk.get("section", ""),
        "breadcrumb":         chunk.get("breadcrumb", ""),

        # Document identity
        "doc_id":             chunk.get("doc_id", ""),
        "doc_title":          chunk.get("doc_title", ""),
        "source_url":         chunk.get("source_url", ""),
        "parent_doc_id":      chunk.get("parent_doc_id", ""),
        "parent_doc_title":   chunk.get("parent_doc_title", ""),
        "root_doc_id":        chunk.get("root_doc_id", ""),
        "root_doc_title":     chunk.get("root_doc_title", ""),
        "depth":              chunk.get("depth", 0),

        # Pipeline metadata
        "token_count":        chunk.get("token_count", 0),
        "created_at":         chunk.get("created_at", ""),
    }

    # image-specific fields
    if chunk_type == "image":
        payload["image_filename"] = chunk.get("image_filename", "")

    return payload


# =============================================================================
# COLLECTION SETUP
# =============================================================================

def setup_collection(client: QdrantClient) -> None:
    """
    Delete the existing collection (if any) and create a fresh one.

    Why delete and recreate instead of upsert?
    On re-ingest, some chunks may be removed (doc deleted, section removed).
    Upsert would leave stale points. Delete + recreate guarantees the
    collection always exactly mirrors the current scraper output.

    Vector config:
        dense  — cosine similarity, 3072d (Gemini gemini-embedding-001)
        sparse — on-disk=False (keep in RAM for fast BM25 lookup)
    """
    existing = {c.name for c in client.get_collections().collections}

    if QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)
        print(f"  🗑  Deleted existing collection: {QDRANT_COLLECTION}")

    client.create_collection(
        collection_name      = QDRANT_COLLECTION,
        vectors_config       = {
            "dense": VectorParams(
                size     = EMBEDDING_DIM,
                distance = Distance.COSINE,
            )
        },
        sparse_vectors_config = {
            "sparse": SparseVectorParams(
                index = SparseIndexParams(on_disk=False)
            )
        },
    )

    print(f"  ✅ Created collection: {QDRANT_COLLECTION}")
    print(f"     Dense:   {EMBEDDING_DIM}d cosine ({QDRANT_COLLECTION})")
    print(f"     Sparse:  BM25-style TF keyword index")
    print(f"     Weights: vector={VECTOR_WEIGHT}, bm25={BM25_WEIGHT}")


# =============================================================================
# PAYLOAD INDEXES
# =============================================================================

def create_payload_indexes(client: QdrantClient) -> None:
    """
    Create payload indexes for fast filtered retrieval.

    Without indexes, filtered queries scan all points.
    With indexes, Qdrant uses an inverted index for O(log n) filtering.

    Indexed fields are those most likely to be used as filters:
        - chunk_type     → filter out section_index / restricted_doc at query time
        - doc_title      → retrieve chunks from a specific document
        - root_doc_title → retrieve chunks from an entire document tree
        - depth          → prefer shallow (root) docs over deep nested ones
        - breadcrumb     → keyword search within breadcrumb path
        - heading_level  → prefer top-level headings for overview questions
        - token_count    → filter out very short or very long chunks
    """
    keyword_fields = [
        "chunk_type",
        "doc_title",
        "root_doc_title",
        "breadcrumb",
        "section",
    ]
    integer_fields = [
        "heading_level",
        "token_count",
        "depth",
    ]

    for field in keyword_fields:
        try:
            client.create_payload_index(
                collection_name = QDRANT_COLLECTION,
                field_name      = field,
                field_schema    = PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            logger.warning(f"Payload index '{field}' skipped: {e}")

    for field in integer_fields:
        try:
            client.create_payload_index(
                collection_name = QDRANT_COLLECTION,
                field_name      = field,
                field_schema    = PayloadSchemaType.INTEGER,
            )
        except Exception as e:
            logger.warning(f"Payload index '{field}' skipped: {e}")

    print(f"  ✅ Payload indexes created")


# =============================================================================
# MAIN
# =============================================================================

def run() -> None:
    """
    Entry point for Stage 3: Uploader.

    Loads embedded chunks, builds Qdrant PointStructs with dense + sparse
    vectors, recreates the collection, uploads all points in batches,
    and runs a quick smoke test to verify retrieval works.
    """
    print("\n" + "═" * 65)
    print("  Stage 3 — Uploader")
    print(f"  Collection: {QDRANT_COLLECTION}")
    print(f"  Reading from: {EMBEDDED_FILE}")
    print("═" * 65)

    # ── Load embedded chunks ──────────────────────────────────────────────
    if not EMBEDDED_FILE.exists():
        print(f"\n  ❌ {EMBEDDED_FILE} not found")
        print(f"     Run embedder first: python -m app.ingest.embedder")
        raise SystemExit(1)

    with open(EMBEDDED_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"\n  Loaded {len(chunks)} chunks")

    # Guard: all chunks must have embeddings before we touch Qdrant
    missing = [c["chunk_id"] for c in chunks if "embedding" not in c]
    if missing:
        print(f"\n  ❌ {len(missing)} chunks missing embeddings")
        print(f"     Run embedder first: python -m app.ingest.embedder")
        raise SystemExit(1)

    # Verify embedding dimension matches config
    sample_emb = next(c for c in chunks if "embedding" in c)
    actual_dim = len(sample_emb["embedding"])
    if actual_dim != EMBEDDING_DIM:
        print(f"\n  ❌ Dimension mismatch: embeddings are {actual_dim}d, config says {EMBEDDING_DIM}d")
        print(f"     Update EMBEDDING_DIM = {actual_dim} in config.py")
        raise SystemExit(1)

    print(f"  ✅ Embeddings verified (dim={actual_dim})")

    # ── Connect to Qdrant ─────────────────────────────────────────────────
    print(f"\n  Connecting to Qdrant...")
    client = get_qdrant_client()

    try:
        client.get_collections()
        print(f"  ✅ Connected")
    except Exception as e:
        print(f"\n  ❌ Cannot connect to Qdrant: {e}")
        if QDRANT_MODE == "server":
            print(f"     Is Qdrant running? docker compose up -d")
        raise SystemExit(1)

    # ── Setup collection ──────────────────────────────────────────────────
    print(f"\n  Setting up collection...")
    setup_collection(client)

    # ── Build points ──────────────────────────────────────────────────────
    print(f"\n  Building {len(chunks)} points...")

    points         = []
    id_collisions  = 0
    seen_point_ids: dict[int, str] = {}

    for chunk in chunks:
        chunk_id  = chunk["chunk_id"]
        point_id  = chunk_id_to_point_id(chunk_id)

        # Log collisions — two different chunks hashing to the same point_id
        # This is extremely rare (~1 in 4B) but worth knowing about
        if point_id in seen_point_ids:
            logger.warning(
                f"Point ID collision: {chunk_id} and "
                f"{seen_point_ids[point_id]} both map to {point_id}"
            )
            id_collisions += 1
        seen_point_ids[point_id] = chunk_id

        # Build sparse vector from embed_text (richer than content alone)
        text_for_sparse              = chunk.get("embed_text", chunk.get("content", ""))
        sparse_indices, sparse_values = build_sparse_vector(text_for_sparse)

        points.append(PointStruct(
            id      = point_id,
            vector  = {
                "dense": chunk["embedding"],
                "sparse": {
                    "indices": sparse_indices,
                    "values":  sparse_values,
                },
            },
            payload = build_payload(chunk),
        ))

    if id_collisions:
        print(f"  ⚠️  Point ID collisions: {id_collisions} (see logs)")

    # ── Upload in batches ─────────────────────────────────────────────────
    print(f"\n  Uploading in batches of {UPLOAD_BATCH_SIZE}...\n")

    for i in tqdm(range(0, len(points), UPLOAD_BATCH_SIZE), desc="  Uploading"):
        batch = points[i : i + UPLOAD_BATCH_SIZE]
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)

    # ── Payload indexes ───────────────────────────────────────────────────
    print(f"\n  Creating payload indexes...")
    create_payload_indexes(client)

    # ── Verify upload ─────────────────────────────────────────────────────
    info           = client.get_collection(QDRANT_COLLECTION)
    uploaded_count = info.points_count
    print(f"\n  ✅ Upload complete — {uploaded_count} points in Qdrant")

    # ── Smoke test ────────────────────────────────────────────────────────
    # Use the first chunk's own embedding to verify retrieval works.
    # Expected: the chunk should be its own top-1 result (score ≈ 1.0).
    print(f"\n  Running smoke test...")
    test_chunk = chunks[0]
    results = client.query_points(
        collection_name = QDRANT_COLLECTION,
        query           = test_chunk["embedding"],
        using           = "dense",
        limit           = 3,
        with_payload    = True,
    ).points

    print(f"  Query: '{test_chunk.get('heading', test_chunk['chunk_id'])[:50]}'")
    print(f"  Top 3 results:")
    for r in results:
        print(
            f"    [{r.score:.3f}] "
            f"{r.payload.get('chunk_type', ''):8s} | "
            f"{r.payload.get('breadcrumb', '')[:55]}"
        )

    # ── Summary ───────────────────────────────────────────────────────────
    type_counts: dict[str, int] = {}
    for chunk in chunks:
        t = chunk.get("chunk_type", "text")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n  {'─' * 40}")
    print(f"  Collection:     {QDRANT_COLLECTION}")
    print(f"  Total points:   {uploaded_count}")
    print(f"  Embedding dim:  {actual_dim}")
    for chunk_type, count in sorted(type_counts.items()):
        print(f"    {chunk_type:<12}: {count}")
    print(f"  Hybrid search:  vector({VECTOR_WEIGHT}) + sparse({BM25_WEIGHT})")
    print("═" * 65)


if __name__ == "__main__":
    from app.ingest.config import validate
    validate()
    run()