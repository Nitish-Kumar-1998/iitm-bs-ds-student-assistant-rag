"""
app.py
------
FastAPI application — the thin HTTP layer over the RAG pipeline.

This file contains NO retrieval or generation logic of its own. It only:
    1. Receives HTTP requests
    2. Calls app.retrieve.run.search() and app.generate.generator.generate_answer()
    3. Streams the result back as Server-Sent Events (SSE)

This preserves the EXACT API contract of the v1 backend so the existing
frontend (Next.js on Vercel) works against this v2 backend with zero
frontend changes:

    POST /ask
        Request:  {"question": str, "history": [{"role": str, "content": str}]}
        Response: SSE stream of events:
            data: {"type": "status",  "text": "..."}
            data: {"type": "token",   "text": "..."}
            data: {"type": "sources", "sources": [...]}
            data: {"type": "images",  "images": [...]}
            data: {"type": "done"}
            data: {"type": "error",   "text": "..."}

    GET /health
        Response: {"status": "ok", ...service info...}

Run:
    uvicorn app.api.app:app --reload --port 8000
"""

import json
import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.api.config import (
    APP_TITLE,
    APP_VERSION,
    CORS_ALLOW_ORIGINS,
    MAX_HISTORY_MESSAGES,
    OFF_TOPIC_PATTERNS,
    OFF_TOPIC_RESPONSE,
    LOG_LEVEL,
    LOG_FORMAT,
)
from app.api.schemas import AskRequest

from app.retrieve.config import QDRANT_COLLECTION
from app.retrieve.retriever import get_qdrant_client
from app.retrieve.run import search

from app.generate.config import GROQ_MODEL
from app.generate.generator import (
    load_prompt,
    build_context,
    build_source_list,
    generate_with_fallback,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("api.app")


# =============================================================================
# APP
# =============================================================================

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = CORS_ALLOW_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# =============================================================================
# SSE HELPER
# =============================================================================

def sse(event_type: str, data: dict | None = None) -> str:
    """
    Format a single Server-Sent Event line.

    Matches the v1 format exactly: {"type": event_type, **data}
    so the frontend's existing event parser needs no changes.
    """
    payload = {"type": event_type}
    if data:
        payload.update(data)
    return f"data: {json.dumps(payload)}\n\n"


# =============================================================================
# OFF-TOPIC DETECTION
# =============================================================================

def is_off_topic(question: str) -> bool:
    """
    Lightweight keyword check for greetings/small talk/unrelated queries.
    Short-circuits before spending a retrieval + generation round-trip
    on messages that obviously don't need it.
    """
    q = question.lower().strip()
    return any(pattern in q for pattern in OFF_TOPIC_PATTERNS)


# =============================================================================
# IMAGE EXTRACTION
# =============================================================================

def extract_images(chunks: list[dict]) -> list[dict]:
    """
    Pull out image-type chunks from the retrieved results so the frontend
    can render them separately from the text answer.

    Mirrors the v1 "images" event — chunk_type == "image" chunks carry
    image_filename (set during ingest from the scraper's OCR metadata).
    """
    images = []
    for chunk in chunks:
        payload = chunk.get("payload", {})
        if payload.get("chunk_type") == "image":
            images.append({
                "file":    payload.get("image_filename", ""),
                "section": payload.get("section", ""),
                "url":     payload.get("source_url", ""),
            })
    return images


# =============================================================================
# /ask — MAIN STREAMING ENDPOINT
# =============================================================================

@app.post("/ask")
async def ask(req: AskRequest):
    """
    Main chat endpoint. Streams a grounded answer via Server-Sent Events.

    Pipeline:
        1. Off-topic check → short-circuit with a friendly redirect
        2. Retrieve  — app.retrieve.run.search()
        3. Generate  — app.generate.generator (context build + Groq streaming)
        4. Stream tokens back as SSE "token" events
        5. Send "sources" and "images" events after the answer completes
        6. Send "done" to signal stream end
    """

    async def stream():
        try:
            question = req.question.strip()

            # ── Off-topic short-circuit ─────────────────────────────────
            if is_off_topic(question):
                yield sse("token", {"text": OFF_TOPIC_RESPONSE})
                yield sse("done")
                return

            # ── Retrieve ─────────────────────────────────────────────────
            yield sse("status", {"text": "Searching programme documents..."})
            chunks = search(question)

            if not chunks:
                yield sse("token", {"text": (
                    "I could not find relevant information in the programme "
                    "documents.\nPlease check the official IITM BS portal: "
                    "https://study.iitm.ac.in/ds/"
                )})
                yield sse("done")
                return

            sources = build_source_list(chunks)
            images  = extract_images(chunks)

            # ── Generate ─────────────────────────────────────────────────
            yield sse("status", {"text": "Generating answer..."})

            system_template = load_prompt()
            context          = build_context(chunks)
            system_prompt    = system_template.format(context=context)

            messages = [{"role": "system", "content": system_prompt}]

            # Forward recent conversation history for follow-up resolution
            for msg in req.history[-MAX_HISTORY_MESSAGES:]:
                messages.append({"role": msg.role, "content": msg.content})

            messages.append({"role": "user", "content": question})

            stream_resp = generate_with_fallback(messages, stream=True)

            for chunk in stream_resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield sse("token", {"text": delta})
                await asyncio.sleep(0)  # yield control to the event loop

            # ── Trailing metadata events ────────────────────────────────
            yield sse("sources", {"sources": sources})
            if images:
                yield sse("images", {"images": images})
            yield sse("done")

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield sse("error", {"text": str(e)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering for SSE
            "Connection":        "keep-alive",
        },
    )


# =============================================================================
# /health — SERVICE STATUS
# =============================================================================

@app.get("/health")
def health():
    """
    Health check — verifies Qdrant is reachable and reports current
    pipeline configuration. Used by Render/uptime monitors and for
    quick manual debugging.
    """
    try:
        client = get_qdrant_client()
        info   = client.get_collection(QDRANT_COLLECTION)
        qdrant_status = f"ok — {info.points_count} points"
    except Exception as e:
        qdrant_status = f"error: {e}"

    return {
        "status":      "ok",
        "version":     APP_VERSION,
        "llm_model":   GROQ_MODEL,
        "qdrant":      qdrant_status,
        "collection":  QDRANT_COLLECTION,
    }


# =============================================================================
# LOCAL DEV ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    from app.api.config import HOST, PORT, validate

    validate()
    uvicorn.run("app.api.app:app", host=HOST, port=PORT, reload=True)
