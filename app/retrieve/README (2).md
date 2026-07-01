# `app/api/` — FastAPI HTTP Layer

The thin HTTP layer over the RAG pipeline. Contains no retrieval or
generation logic of its own — only request handling, SSE streaming, and
wiring to `app.retrieve` and `app.generate`.

---

## What it does

| File | What it contains |
|------|-------------------|
| `config.py` | App title, host/port, CORS, history limit, off-topic patterns |
| `schemas.py` | Pydantic request models (`ChatMessage`, `AskRequest`) |
| `app.py` | FastAPI app, `/ask` (SSE streaming) and `/health` endpoints |

---

## API contract — unchanged from v1

This backend is a **drop-in replacement** for the v1 API. The existing
Next.js frontend works against it with zero changes.

### `POST /ask`

**Request:**
```json
{
  "question": "what is the OPPE exam",
  "history": [
    {"role": "user", "content": "previous question"},
    {"role": "assistant", "content": "previous answer"}
  ]
}
```

**Response:** Server-Sent Events stream, `Content-Type: text/event-stream`

```
data: {"type": "status", "text": "Searching programme documents..."}

data: {"type": "status", "text": "Generating answer..."}

data: {"type": "token", "text": "The"}

data: {"type": "token", "text": " OPPE"}

data: {"type": "token", "text": " exam..."}

data: {"type": "sources", "sources": [{"index": 1, "doc_title": "...", "breadcrumb": "...", "source_url": "...", "chunk_type": "text"}]}

data: {"type": "images", "images": [{"file": "...", "section": "...", "url": "..."}]}

data: {"type": "done"}
```

On error:
```
data: {"type": "error", "text": "error message"}
```

### `GET /health`

```json
{
  "status": "ok",
  "version": "2.0.0",
  "llm_model": "llama-3.3-70b-versatile",
  "qdrant": "ok — 914 points",
  "collection": "iitm_bs_docs_chunks"
}
```

---

## Pipeline flow per request

```
POST /ask
  │
  ├─→ is_off_topic()?  → yes: short-circuit with friendly redirect, done
  │
  ├─→ app.retrieve.run.search(question)
  │     → hybrid search (dense + sparse) → RRF fusion → cross-encoder rerank
  │     → top 5 chunks
  │
  ├─→ build_source_list(chunks)  → citation metadata for frontend
  ├─→ extract_images(chunks)     → image chunks for frontend
  │
  ├─→ app.generate.generator
  │     → load_prompt() + build_context(chunks) → system prompt
  │     → generate_with_fallback() → Groq streaming response
  │
  ├─→ stream "token" events as Groq generates
  ├─→ stream "sources" event
  ├─→ stream "images" event (if any)
  └─→ stream "done" event
```

---

## Run

**Local development:**
```bash
uvicorn app.api.app:app --reload --port 8000
```

**Validate config (checks retrieve + generate stages too):**
```bash
python -m app.api.config
```

**Test the endpoint:**
```bash
curl -N -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "what is the OPPE exam", "history": []}'

curl http://localhost:8000/health
```

---

## Configuration

| Setting | Value | Why |
|---------|-------|-----|
| `CORS_ALLOW_ORIGINS` | `["*"]` | Open for now — matches v1; tighten to specific Vercel URL once stable |
| `MAX_HISTORY_MESSAGES` | 6 | Recent turns forwarded for follow-up question resolution |
| Rate limiting | None | Intentionally not added — keep it simple for now |

---

## Prerequisites

1. Full ingest pipeline must have run: `python -m app.ingest.run`
2. Retrieve stage must be working: `python -m app.retrieve.run "test"`
3. Generate stage must be working: `python -m app.generate.run "test"`
4. `.env` must have `GEMINI_API_KEY` and `GROQ_API_KEY` set

---

## Deployment notes

For production (Render), the start command becomes:
```bash
uvicorn app.api.app:app --host 0.0.0.0 --port $PORT
```

Switch `QDRANT_MODE=server` in `.env` and point `QDRANT_URL` /
`QDRANT_API_KEY` at your production Qdrant instance before deploying —
this codebase currently defaults to `QDRANT_MODE=embedded` for local
development, which will NOT work on Render (ephemeral filesystem).
