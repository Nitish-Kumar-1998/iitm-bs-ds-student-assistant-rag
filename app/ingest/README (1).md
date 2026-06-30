# `app/ingest/` — Ingest Pipeline

Converts scraped markdown documents into searchable Qdrant vectors.

---

## What it does

| Stage | File | Input | Output |
|-------|------|-------|--------|
| 1 | `chunker.py` | `app/scraper/data/docs/**/*.md` | `data/chunks.json` |
| 2 | `embedder.py` | `data/chunks.json` | `data/chunks_embedded.json` |
| 3 | `uploader.py` | `data/chunks_embedded.json` | Qdrant collection |

---

## Chunk types

| Type | Description |
|------|-------------|
| `text` | Prose content — two-pass split (header boundary + size enforcement) |
| `table` | Markdown tables converted to plain text for reranker compatibility |
| `image` | OCR text extracted from `[IMAGE: ...]` blocks by the scraper |

---

## Run

**Full pipeline (recommended):**
```bash
python -m app.ingest.run
```

**Individual stages:**
```bash
python -m app.ingest.chunker     # Stage 1 only
python -m app.ingest.embedder    # Stage 2 only
python -m app.ingest.uploader    # Stage 3 only
```

**Skip stages (when resuming):**
```bash
python -m app.ingest.run --skip-chunking     # start from Stage 2
python -m app.ingest.run --skip-embedding    # start from Stage 3 only
```

**Validate config:**
```bash
python -m app.ingest.config
```

---

## Configuration

All settings are in `config.py`. Key values:

| Setting | Value | Why |
|---------|-------|-----|
| `CHUNK_SIZE` | 800 tokens | IITM docs have rich tables and long FAQ sections |
| `CHUNK_OVERLAP` | 100 tokens | Prevents losing sentences at chunk boundaries |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Free tier, 3072d, strong multilingual |
| `QDRANT_COLLECTION` | `iitm_bs_docs_chunks` | Explicit name — content is clear |
| `VECTOR_WEIGHT` | 0.7 | Semantic search is primary |
| `BM25_WEIGHT` | 0.3 | Keyword matching as secondary signal |

---

## Chunk schema

Every chunk carries:

```json
{
  "chunk_id":         "a3f9b1c2d4e5",   // stable MD5 hash of content
  "chunk_type":       "text",            // text | table | image
  "content":          "...",             // actual text sent to LLM
  "embed_text":       "...",             // heading + breadcrumb + content (richer vector)
  "heading":          "Fee Structure",
  "heading_level":    2,
  "section":          "Fee Structure",
  "breadcrumb":       "Student Handbook > Fees > Fee Structure",
  "doc_id":           "9ae4dc8b",
  "doc_title":        "IITM BS Degree Programme - Student Handbook",
  "source_url":       "https://docs.google.com/...",
  "parent_doc_id":    null,
  "parent_doc_title": "",
  "root_doc_id":      "9ae4dc8b",
  "root_doc_title":   "IITM BS Degree Programme - Student Handbook",
  "depth":            0,
  "token_count":      312,
  "created_at":       "2026-06-29T17:00:00+00:00",
  "version":          "2"
}
```

---

## Prerequisites

1. Scraper must have run first: `python -m app.scraper.scraper`
2. Qdrant must be running: `docker compose up -d`
3. `.env` must have `GEMINI_API_KEY` set

---

## Resume support

If embedding is interrupted (network error, quota hit), re-run `embedder.py`.
It will skip already-embedded chunks and continue from where it stopped.
Progress is saved every 10 chunks to `data/embedding_progress.json`.

---

## Output files (gitignored)

```
data/
├── chunks.json               # Stage 1 output — raw chunks
├── chunks_embedded.json      # Stage 2 output — chunks + embeddings
└── embedding_progress.json   # Resume checkpoint (deleted on success)
```
