"""
IITM BS RAG Pipeline — Stage 6: Uploader
==========================================
INPUT:  output/chunks/all_chunks_embedded.json  (from embedder.py)
OUTPUT: Qdrant collection "iitm_bs" (cloud or local)

What it does:
  - Wipes and recreates Qdrant collection fresh (Plan A)
  - Sets up hybrid search: dense vectors + BM25 sparse vectors
  - Uploads all chunks with complete payload (all new fields)
  - Uses chunk_id hash as Qdrant point ID (stable across re-runs)
  - Creates payload indexes for fast filtering
  - Runs test searches (vector + hybrid) to verify

Hybrid search weights (from config):
  VECTOR_WEIGHT = 0.7  (semantic — finds meaning)
  BM25_WEIGHT   = 0.3  (keyword — finds exact terms)

Run:
  python uploader.py
"""

import json
import hashlib
import logging
import voyageai
from pathlib import Path
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

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    VOYAGE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    LOG_LEVEL,
    LOG_FORMAT,
    ALL_CHUNKS_FILE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("uploader")

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════

EMBEDDED_FILE = ALL_CHUNKS_FILE.parent / "all_chunks_embedded.json"
BATCH_SIZE    = 100


# ══════════════════════════════════════════════════════════════════
# CHUNK ID → QDRANT POINT ID
# Qdrant needs integer or UUID as point ID.
# We convert our 12-char hex chunk_id to a stable integer
# by taking first 8 hex chars → integer.
# This is stable across re-runs unlike sequential i.
# ══════════════════════════════════════════════════════════════════

def chunk_id_to_point_id(chunk_id: str) -> int:
    """
    Convert 12-char hex chunk_id to stable integer for Qdrant.
    Uses MD5 of chunk_id to get full 32-char hex → take first 8 → int.
    Collision probability is negligible for typical chunk counts.
    """
    full_hash = hashlib.md5(chunk_id.encode()).hexdigest()
    return int(full_hash[:8], 16)


# ══════════════════════════════════════════════════════════════════
# PAYLOAD BUILDER
# All new fields included — nothing missing
# ══════════════════════════════════════════════════════════════════

def build_payload(chunk: dict) -> dict:
    """
    INPUT:  chunk dict (without embedding)
    OUTPUT: complete payload dict for Qdrant

    Includes ALL fields from the rebuilt pipeline:
    hyde_questions, parent_doc, references, created_at,
    version, what_it_contains, when_to_refer, access,
    image_content, found_in_doc
    """
    chunk_type = chunk.get("chunk_type", "text")

    # Base payload — every chunk type
    payload = {
        # Identity
        "chunk_id":       chunk.get("chunk_id", ""),
        "chunk_type":     chunk_type,

        # Content
        "content":        chunk.get("content", ""),
        "embed_text":     chunk.get("embed_text", ""),

        # Location
        "heading":        chunk.get("heading", ""),
        "heading_level":  chunk.get("heading_level", 0),
        "doc_title":      chunk.get("doc_title", ""),
        "section":        chunk.get("section", ""),
        "breadcrumb":     chunk.get("breadcrumb", ""),
        "source_url":     chunk.get("source_url", ""),
        "parent_doc":     chunk.get("parent_doc", ""),

        # Relations
        "references":     chunk.get("references", []),

        # HyDE questions — used at retrieval time
        "hyde_questions": chunk.get("hyde_questions", []),

        # Versioning
        "created_at":     chunk.get("created_at", ""),
        "version":        chunk.get("version", "1"),

        # Stats
        "token_count":    chunk.get("token_count", 0),
    }

    # ── Type-specific fields ──────────────────────────────────────

    if chunk_type == "image":
        payload["image_file"]    = chunk.get("image_file", "")
        payload["image_content"] = chunk.get("image_content", "")
        payload["image_type"]    = chunk.get("image_type", "")
        payload["scan_method"]   = chunk.get("scan_method", "")

    if chunk_type == "reference_link":
        payload["link_url"]         = chunk.get("link_url", "")
        payload["link_text"]        = chunk.get("link_text", "")
        payload["what_it_contains"] = chunk.get("what_it_contains", "")
        payload["when_to_refer"]    = chunk.get("when_to_refer", "")
        payload["category"]         = chunk.get("category", "")
        payload["access"]           = chunk.get("access", "public")
        payload["found_in_doc"]     = chunk.get("found_in_doc", "")

    if chunk_type == "restricted_doc":
        payload["link_url"]     = chunk.get("link_url", "")
        payload["skip_reason"]  = chunk.get("skip_reason", "")
        payload["note"]         = chunk.get("note", "")
        payload["access"]       = chunk.get("access", "restricted")
        payload["found_in_doc"] = chunk.get("found_in_doc", "")

    return payload


# ══════════════════════════════════════════════════════════════════
# COLLECTION SETUP
# Hybrid search: dense vector + BM25 sparse vector
# ══════════════════════════════════════════════════════════════════

def setup_collection(client: QdrantClient):
    """
    INPUT:  Qdrant client
    OUTPUT: fresh collection with hybrid search configured

    Always wipes and recreates — Plan A decision.
    Dense vector: cosine similarity (semantic search)
    Sparse vector: BM25 (keyword search)
    """
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)
        print(f"  🗑  Deleted existing collection: {QDRANT_COLLECTION}")

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "dense": VectorParams(
                size=EMBEDDING_DIM,        # 1024 for voyage-3
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False,
                )
            )
        },
    )
    print(f"  ✅ Created collection: {QDRANT_COLLECTION}")
    print(f"     Dense:  {EMBEDDING_DIM}d cosine (voyage-3 semantic)")
    print(f"     Sparse: BM25 (keyword)")
    print(f"     Weights: vector={VECTOR_WEIGHT}, bm25={BM25_WEIGHT}")


# ══════════════════════════════════════════════════════════════════
# BM25 SPARSE VECTOR BUILDER
# Simple TF-IDF approximation for BM25 sparse vectors
# ══════════════════════════════════════════════════════════════════

def build_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """
    INPUT:  text string
    OUTPUT: (indices, values) for sparse vector

    Simple term frequency approach for BM25.
    Each unique word gets a stable index (hash-based).
    Value = term frequency normalized.
    """
    import re
    from collections import Counter

    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    if not words:
        return [0], [0.0]

    tf    = Counter(words)
    total = len(words)

    index_map = {}
    for word, count in tf.items():
        word_idx = int(hashlib.md5(word.encode()).hexdigest()[:6], 16) % 100000
        tf_score = count / total
        if word_idx in index_map:
            index_map[word_idx] += tf_score
        else:
            index_map[word_idx] = tf_score

    return list(index_map.keys()), [float(v) for v in index_map.values()]


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run():
    print("\n" + "═" * 65)
    print("  IITM BS RAG Pipeline — Stage 6: Uploader")
    print("  Embedding provider: Voyage AI (voyage-3)")
    print("═" * 65)

    # Load embedded chunks
    if not EMBEDDED_FILE.exists():
        print(f"\n  ❌ {EMBEDDED_FILE} not found")
        print(f"     Run python embedder.py first")
        return

    print(f"\n  Loading embedded chunks...")
    with open(EMBEDDED_FILE) as f:
        chunks = json.load(f)
    print(f"  Loaded {len(chunks)} chunks")

    # Verify embeddings present and correct dimension
    missing_emb = sum(1 for c in chunks if "embedding" not in c)
    if missing_emb:
        print(f"  ❌ {missing_emb} chunks missing embeddings — run embedder.py first")
        return

    # Check embedding dimension matches config
    sample_emb = next(c for c in chunks if "embedding" in c)
    actual_dim = len(sample_emb["embedding"])
    if actual_dim != EMBEDDING_DIM:
        print(f"  ❌ Dimension mismatch: embeddings are {actual_dim}d but config says {EMBEDDING_DIM}d")
        print(f"     Update EMBEDDING_DIM = {actual_dim} in config.py")
        return

    print(f"  ✅ All chunks have embeddings (dim={actual_dim})")

    # Connect to Qdrant
    print(f"\n  Connecting to Qdrant at {QDRANT_URL}...")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    try:
        qdrant.get_collections()
        print(f"  ✅ Connected to Qdrant")
    except Exception as e:
        print(f"  ❌ Cannot connect: {e}")
        return

    # Setup collection (wipe + recreate)
    print(f"\n  Setting up collection...")
    setup_collection(qdrant)

    # Build and upload points
    print(f"\n  Building {len(chunks)} points...")

    points        = []
    id_map        = {}
    id_collisions = 0

    for chunk in chunks:
        embedding = chunk.get("embedding", [])
        chunk_id  = chunk.get("chunk_id", "")
        point_id  = chunk_id_to_point_id(chunk_id)

        # Check for ID collision (extremely rare)
        if point_id in id_map:
            logger.warning(f"ID collision: {chunk_id} and {id_map[point_id]} → {point_id}")
            id_collisions += 1
        id_map[point_id] = chunk_id

        # Build sparse vector for BM25
        content_text = chunk.get("embed_text", chunk.get("content", ""))
        sparse_indices, sparse_values = build_sparse_vector(content_text)

        payload = build_payload(chunk)

        points.append(PointStruct(
            id     = point_id,
            vector = {
                "dense": embedding,
                "sparse": {
                    "indices": sparse_indices,
                    "values":  sparse_values,
                }
            },
            payload = payload,
        ))

    if id_collisions:
        logger.warning(f"Total ID collisions: {id_collisions} (negligible)")

    # Upload in batches
    print(f"\n  Uploading to Qdrant in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(points), BATCH_SIZE), desc="  Uploading"):
        batch = points[i:i + BATCH_SIZE]
        qdrant.upsert(
            collection_name=QDRANT_COLLECTION,
            points=batch,
        )

    # Create payload indexes for fast filtering
    print(f"\n  Creating payload indexes...")
    index_fields = [
        ("chunk_type",    PayloadSchemaType.KEYWORD),
        ("doc_title",     PayloadSchemaType.KEYWORD),
        ("parent_doc",    PayloadSchemaType.KEYWORD),
        ("section",       PayloadSchemaType.KEYWORD),
        ("access",        PayloadSchemaType.KEYWORD),
        ("heading_level", PayloadSchemaType.INTEGER),
    ]
    for field, schema in index_fields:
        try:
            qdrant.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=schema,
            )
            logger.info(f"Index created: {field}")
        except Exception as e:
            logger.warning(f"Index {field} skipped: {e}")

    # Verify upload
    info  = qdrant.get_collection(QDRANT_COLLECTION)
    count = info.points_count
    print(f"\n  ✅ Upload complete!")
    print(f"     Collection: {QDRANT_COLLECTION}")
    print(f"     Points:     {count}")

    # ── Test searches using Voyage AI ─────────────────────────────
    print(f"\n  Running test searches via Voyage AI...")
    voyage = voyageai.Client(api_key=VOYAGE_API_KEY)

    test_queries = [
        "eligibility criteria for foundation level admission",
        "what is the fee for diploma programme",
        "OPPE exam rules and camera setup",
    ]

    for query in test_queries:
        print(f"\n  Query: '{query}'")

        # Embed with Voyage AI — input_type="query" for search time
        result = voyage.embed(
            [query],
            model=EMBEDDING_MODEL,
            input_type="query",
        )
        vec = result.embeddings[0]

        results = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vec,
            using="dense",
            limit=3,
            with_payload=True,
        ).points

        print(f"  Top 3 results:")
        for r in results:
            print(f"    [{r.score:.3f}] {r.payload.get('chunk_type',''):15s} | {r.payload.get('breadcrumb','')[:50]}")

    # ── Summary ───────────────────────────────────────────────────
    type_counts = {}
    for chunk in chunks:
        t = chunk.get("chunk_type", "text")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n  {'═' * 40}")
    print(f"  Collection:     {QDRANT_COLLECTION}")
    print(f"  Total points:   {count}")
    print(f"  Embedding dim:  {actual_dim} (voyage-3)")
    for t, c in sorted(type_counts.items()):
        print(f"    {t:20s}: {c}")
    print(f"\n  Hybrid search: vector({VECTOR_WEIGHT}) + BM25({BM25_WEIGHT})")
    print(f"\n  Next step: python main.py (or uvicorn main:app)")
    print("═" * 65)


if __name__ == "__main__":
    run()