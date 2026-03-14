"""
IITM BS RAG Pipeline — Stage 11: FastAPI Backend
=================================================
Changes in this version:
  - Replaced local SentenceTransformer + CrossEncoder with Voyage AI API
  - Zero RAM overhead for ML models (voyage-3 + rerank-2)
  - Faster startup (no model loading)
  - Better retrieval quality (voyage-3 vs MiniLM)

Endpoints:
  POST /ask        → streaming SSE answer
  GET  /health     → check all services are up

Run:
  uvicorn main:app --reload --port 8000
"""

import os
import json
import hashlib
import asyncio
import logging
import time
import voyageai
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_FALLBACK_MODELS,
    LLM_PROVIDER,
    VOYAGE_API_KEY,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    RERANKER_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    RAG_SYSTEM_PROMPT,
    STREAM_RESPONSE,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("main")

MAX_HISTORY = 6

# increased retrieval pool for wider net before reranking
RETRIEVAL_TOP_K_EXPANDED = 20

# confidence threshold — below this score, broaden search
CONFIDENCE_THRESHOLD = 0.3

# ══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are the IITM BS Programme Assistant — an expert on the
IIT Madras Online BS Degree Programme in Data Science.

## YOUR KNOWLEDGE SOURCE
Answer ONLY from the context chunks provided below.
Never guess or make up policy details, fees, dates, or eligibility criteria.

## HANDLING DIFFERENT QUESTION TYPES

OFF-TOPIC (not about IITM BS):
→ Politely redirect:
  "I can only help with IITM BS programme questions. I can help you with:
   courses, fees, exams, projects, credit transfer, degree requirements.
   What would you like to know?"

VAGUE QUESTION (context has partial answer):
→ Give what you found, then ask ONE specific follow-up.
  Only mention courses/topics that actually appear in the retrieved context.

VAGUE QUESTION (no useful context found):
→ Ask ONE clarifying question.

SPECIFIC QUESTION:
→ Answer directly and completely from context.

FOLLOW-UP QUESTION:
→ Use conversation history to understand what "it", "that", "this" refers to.
→ Combine history + current question for a complete answer.

QUESTION ABOUT RESTRICTED DOCUMENT:
→ Tell student the document exists and provide the URL.

## FORMAT RULES
- Simple fact → 1-3 sentences, then detail if needed
- Eligibility/criteria → bullet list
- Process/steps → numbered list
- Comparison → markdown table
- Use **bold** for deadlines, marks, fees, important terms
- Use `code` for course codes like `BSCS1002`
- Never pad with filler ("Great question!", "Certainly!", etc.)
- NEVER mention "CONTEXT 1", "CONTEXT 2" etc. — internal labels, never visible to students

## FIX 4 — DIRECT ANSWER RULE
- Questions starting with "Can I", "Is there", "Does", "Will", "Am I" MUST start with YES or NO
- Never hedge with "it seems", "it appears", "however", "based on the context"
- If you know the answer from context → state it directly and confidently
- If context only partially answers → give what you know, then state exactly what is missing
- Never say "there is no information" if there is even partial relevant context

## FIX 6 — CONFIDENCE RULE
- If you are certain from context → answer directly, no qualifiers
- If context is partial → say "Based on available documents: [answer]. For complete details check: [URL]"
- Never use phrases like "it seems", "it appears", "it is possible", "may be", "might be" unless genuinely uncertain
- The programme IS fully online and self-paced — state this confidently when asked about flexibility

## CITATIONS — MANDATORY URL REQUIRED
End EVERY answer with source citation in this exact format:

📎 **Source:** [section name] — [document title] 🔗 [URL]

RULES:
- URL is MANDATORY — never omit it
- If context has a source_url → use it exactly
- If no URL in context → use https://study.iitm.ac.in/ds/
- If multiple sources → list each on its own line with its own URL
- NEVER include internal labels like [SOURCE:...] or [CONTEXT N] in citations
- Use actual document title and section name only

## LINKS
If context contains a reference link with a URL — always include it inline in your answer.
If context has when_to_refer field — use it to decide if link is relevant.

## IMAGES
If context contains an image chunk — mention it:
"As shown in the diagram: [description of what image shows]"

## CANNOT ANSWER
If answer truly not in context:
"I could not find this in the current programme documents.
 Please check: https://study.iitm.ac.in/ds/"
"""

# ══════════════════════════════════════════════════════════════════
# INIT — Voyage AI client (no local models, zero RAM overhead)
# ══════════════════════════════════════════════════════════════════

print("Initialising Voyage AI client...")
voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
print(f"✅ Voyage AI ready: {EMBEDDING_MODEL} + {RERANKER_MODEL}")

print(f"Connecting to Qdrant at {QDRANT_URL}...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
print("✅ Qdrant connected")

print(f"Initialising LLM client ({LLM_PROVIDER})...")
llm_client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    timeout=20,
    max_retries=0,
)
print(f"✅ LLM client ready: {LLM_MODEL}")
print("✅ Backend ready\n")

# ══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(title="IITM BS RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []


# ══════════════════════════════════════════════════════════════════
# LLM CALLER WITH FALLBACK + BACKOFF
# ══════════════════════════════════════════════════════════════════

def llm_call(messages: list, max_tokens: int = 200, stream: bool = False):
    models_to_try = [LLM_MODEL] + LLM_FALLBACK_MODELS
    last_error    = None

    for i, model in enumerate(models_to_try):
        try:
            if i > 0:
                wait = min(2 ** i, 8)
                logger.info(f"Waiting {wait}s before trying {model}...")
                time.sleep(wait)

            resp = llm_client.chat.completions.create(
                model       = model,
                messages    = messages,
                max_tokens  = max_tokens,
                temperature = 0.1,
                stream      = stream,
            )
            if model != LLM_MODEL:
                logger.info(f"Using fallback model: {model}")
            return resp

        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in [
                "rate_limit", "429", "decommissioned",
                "model_not_found", "timed out", "timeout",
                "connection", "read timed out"
            ]):
                logger.warning(f"Model {model} failed ({type(e).__name__}), trying next...")
                last_error = e
                continue
            raise e

    raise Exception(f"All LLM models exhausted. Last error: {last_error}")


# ══════════════════════════════════════════════════════════════════
# QUERY CLASSIFIER
# ══════════════════════════════════════════════════════════════════

OFF_TOPIC_PATTERNS = [
    "who are you", "what are you", "what can you do", "what do you know",
    "how do you work", "are you an ai", "are you a bot",
    "hello", "hi ", "hey ", "thanks", "thank you", "bye", "goodbye",
    "what is the capital", "who is the president", "weather",
    "recipe", "movie", "song", "game",
]

def is_off_topic(question: str) -> bool:
    q = question.lower().strip()
    return any(pattern in q for pattern in OFF_TOPIC_PATTERNS)


# ══════════════════════════════════════════════════════════════════
# QUERY EXPANSION — rewrite all questions for better retrieval
# ══════════════════════════════════════════════════════════════════

def rewrite_query(question: str, history: list[ChatMessage]) -> str:
    """
    Rewrite ALL questions for better retrieval, not just vague ones.
    - Expands synonyms (fail → not pass, academic failure)
    - Resolves pronouns using history
    - Makes implicit topics explicit
    - Short questions get expanded to full searchable queries
    """
    history_text = ""
    if history:
        history_text = "\n".join(
            f"{m.role}: {m.content}" for m in history[-2:]
        )

    try:
        prompt = (
            f"Rewrite this student question to be specific and highly searchable "
            f"in an IITM BS programme policy document.\n\n"
            f"Rules:\n"
            f"- Expand 'fail exam' → 'fail course score below 40 not clear repeat attempt academic performance consequences'\n"
            f"- Expand 'fee discount' → 'fee waiver concession SC ST OBC EWS income poverty'\n"
            f"- Expand 'how long' → 'duration minimum maximum years terms complete degree'\n"
            f"- Expand 'work job' → 'self-paced online flexible working professionals part time'\n"
            f"- Expand 'grading document' → 'grading formula marks assignment quiz OPPE weightage'\n"
            f"- For 'what happens if' questions: include subject and consequence explicitly\n"
            f"- Always include the specific topic, level, or course name\n"
            f"- Return ONLY the rewritten question, nothing else\n\n"
        )
        if history_text:
            prompt += f"Conversation so far:\n{history_text}\n\n"
        prompt += f"Question to rewrite: {question}"

        resp = llm_call(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        rewritten = resp.choices[0].message.content.strip()
        logger.info(f"Rewritten: '{question}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"Rewrite failed: {e}")
        return question


# ══════════════════════════════════════════════════════════════════
# SPARSE VECTOR BUILDER (BM25 approximation)
# ══════════════════════════════════════════════════════════════════

def build_sparse_vector(text: str) -> tuple[list[int], list[float]]:
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
# HYBRID RETRIEVAL — Voyage AI embed + rerank
# ══════════════════════════════════════════════════════════════════

def retrieve_chunks(question: str, top_k: int = RETRIEVAL_TOP_K_EXPANDED) -> tuple[list[dict], float]:
    """
    Returns (chunks, top_rerank_score) so caller can check confidence.
    Uses Voyage AI for both embedding and reranking — zero local RAM.
    """

    # ── Step 1: Embed query with Voyage AI ──────────────────────
    embed_result = voyage_client.embed(
        [question],
        model=EMBEDDING_MODEL,
        input_type="query",           # "query" for search time, "document" for upload time
    )
    query_vec = embed_result.embeddings[0]

    # ── Step 2: Build sparse vector for BM25 ────────────────────
    sparse_indices, sparse_values = build_sparse_vector(question)

    # ── Step 3: Dense search ────────────────────────────────────
    try:
        dense_results = qdrant.query_points(
            collection_name = QDRANT_COLLECTION,
            query           = query_vec,
            using           = "dense",
            limit           = top_k,
            with_payload    = True,
        ).points
    except Exception as e:
        logger.warning(f"Dense search failed: {e}, falling back to basic search")
        dense_results = qdrant.query_points(
            collection_name = QDRANT_COLLECTION,
            query           = query_vec,
            limit           = top_k,
            with_payload    = True,
        ).points

    # ── Step 4: Sparse search (BM25) ────────────────────────────
    try:
        from qdrant_client.models import SparseVector
        sparse_results = qdrant.query_points(
            collection_name = QDRANT_COLLECTION,
            query           = SparseVector(indices=sparse_indices, values=sparse_values),
            using           = "sparse",
            limit           = top_k,
            with_payload    = True,
        ).points
    except Exception as e:
        logger.warning(f"Sparse search failed: {e}")
        sparse_results = []

    # ── Step 5: Merge dense + sparse scores ─────────────────────
    seen_ids = {}
    for point in dense_results:
        seen_ids[point.id] = {"point": point, "score": point.score * VECTOR_WEIGHT}
    for point in sparse_results:
        if point.id in seen_ids:
            seen_ids[point.id]["score"] += point.score * BM25_WEIGHT
        else:
            seen_ids[point.id] = {"point": point, "score": point.score * BM25_WEIGHT}

    merged     = sorted(seen_ids.values(), key=lambda x: x["score"], reverse=True)
    top_points = [m["point"] for m in merged[:top_k]]

    if not top_points:
        return [], 0.0

    # ── Step 6: Cross-reference expansion ───────────────────────
    all_points   = list(top_points)
    ref_sections = set()
    for point in top_points:
        refs = point.payload.get("references", [])
        for ref in refs[:3]:
            ref_sections.add(ref)

    if ref_sections:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            ref_results = qdrant.query_points(
                collection_name = QDRANT_COLLECTION,
                query           = query_vec,
                using           = "dense",
                query_filter    = Filter(
                    must=[FieldCondition(
                        key   = "section",
                        match = MatchAny(any=list(ref_sections))
                    )]
                ),
                limit        = 5,
                with_payload = True,
            ).points
            existing_ids = {p.id for p in all_points}
            for p in ref_results:
                if p.id not in existing_ids:
                    all_points.append(p)
                    existing_ids.add(p.id)
        except Exception as e:
            logger.warning(f"Cross-ref fetch failed: {e}")

    # ── Step 7: Rerank with Voyage AI ───────────────────────────
    documents = [p.payload.get("content", "") for p in all_points]

    rerank_result = voyage_client.rerank(
        query=question,
        documents=documents,
        model=RERANKER_MODEL,
        top_k=RERANK_TOP_K,
    )

    top_score = rerank_result.results[0].relevance_score if rerank_result.results else 0.0

    chunks = []
    for item in rerank_result.results:
        point   = all_points[item.index]       # voyage returns original index
        payload = dict(point.payload)
        payload["rerank_score"] = float(item.relevance_score)
        chunks.append(payload)

    return chunks, top_score


# ══════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════

def build_context(chunks: list[dict]) -> tuple[str, list[dict], list[dict]]:
    context_parts = []
    sources       = []
    images        = []
    seen_sources  = set()

    for i, chunk in enumerate(chunks):
        chunk_type = chunk.get("chunk_type", "text")
        content    = chunk.get("content", "")
        breadcrumb = chunk.get("breadcrumb", "")
        doc_title  = chunk.get("doc_title", "")
        section    = chunk.get("section", "")
        source_url = chunk.get("source_url", "https://study.iitm.ac.in/ds/")

        # noise section filter — skip generic handbook boilerplate as sources
        noise_sections = [
            "this will be in effect",
            "important advisory",
            "section index",
        ]
        is_noise = any(n in section.lower() for n in noise_sections)

        if chunk_type in ("text", "table", "section_index"):
            context_parts.append(
                f"[SOURCE: {doc_title} — {section} | URL: {source_url}]\n{content}"
            )
            source_key   = f"{doc_title}|{section}"
            rerank_score = chunk.get("rerank_score", 0)
            # only add to sources if high confidence AND not noise section
            if source_key not in seen_sources and rerank_score > 0.4 and not is_noise:
                seen_sources.add(source_key)
                sources.append({
                    "doc":     doc_title,
                    "section": section,
                    "url":     source_url,
                    "type":    "document",
                })

        elif chunk_type == "image":
            image_content = chunk.get("image_content", "")
            context_parts.append(
                f"[IMAGE: {section} | URL: {source_url}]\n"
                f"Image file: {chunk.get('image_file','')}\n"
                f"Content: {image_content or content}"
            )
            if chunk.get("image_file"):
                images.append({
                    "file":    chunk["image_file"],
                    "section": section,
                    "url":     source_url,
                })
            source_key = f"{doc_title}|{section}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "doc":     doc_title,
                    "section": section,
                    "url":     source_url,
                    "type":    "image",
                })

        elif chunk_type == "reference_link":
            when_to_refer    = chunk.get("when_to_refer", "")
            what_it_contains = chunk.get("what_it_contains", "")
            link_url         = chunk.get("link_url", source_url)
            context_parts.append(
                f"[REFERENCE LINK: {section} | URL: {link_url}]\n"
                f"Title: {chunk.get('link_text','')}\n"
                f"URL: {link_url}\n"
                f"Contains: {what_it_contains}\n"
                f"Use when: {when_to_refer}"
            )
            sources.append({
                "doc":              doc_title,
                "section":          section,
                "url":              link_url,
                "text":             chunk.get("link_text", ""),
                "what_it_contains": what_it_contains,
                "type":             "link",
                "is_link":          True,
            })

        elif chunk_type == "restricted_doc":
            link_url = chunk.get("link_url", source_url)
            context_parts.append(
                f"[RESTRICTED DOCUMENT: {section} | URL: {link_url}]\n"
                f"URL: {link_url}\n"
                f"Note: {chunk.get('note', 'Requires IITM login to access.')}\n"
                f"Context: {content}"
            )
            sources.append({
                "doc":     doc_title,
                "section": section,
                "url":     link_url,
                "type":    "restricted",
                "note":    chunk.get("note", ""),
            })

    return "\n\n".join(context_parts), sources, images


# ══════════════════════════════════════════════════════════════════
# SSE HELPER
# ══════════════════════════════════════════════════════════════════

def sse(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


# ══════════════════════════════════════════════════════════════════
# ASK ENDPOINT
# ══════════════════════════════════════════════════════════════════

@app.post("/ask")
async def ask(req: AskRequest):

    async def stream():
        try:
            question = req.question.strip()

            # 1. Off-topic check
            if is_off_topic(question):
                yield sse("token", {"text": (
                    "I'm the IITM BS Programme Assistant. I can help you with:\n\n"
                    "- 📚 Courses and registration\n"
                    "- 💰 Fees and payments\n"
                    "- 📝 Exams (quizzes, OPPE, end-term)\n"
                    "- 🎯 Projects (MLP, BDM, App Dev)\n"
                    "- 🔄 Credit transfer (NPTEL, SWAYAM)\n"
                    "- 🎓 Degree requirements and pathways\n\n"
                    "What would you like to know about the IITM BS programme?"
                )})
                yield sse("done", {})
                return

            # 2. Rewrite ALL questions for better retrieval
            yield sse("status", {"text": "Understanding your question..."})
            rewritten = rewrite_query(question, req.history)

            # 3. Hybrid retrieval + Voyage AI rerank
            yield sse("status", {"text": "Searching programme documents..."})
            chunks, top_score = retrieve_chunks(rewritten)

            # confidence threshold — broaden search if score too low
            if top_score < CONFIDENCE_THRESHOLD and chunks:
                logger.info(f"Low confidence ({top_score:.2f}), broadening search with original question...")
                yield sse("status", {"text": "Searching deeper..."})
                chunks2, score2 = retrieve_chunks(question, top_k=30)
                if score2 > top_score:
                    chunks    = chunks2
                    top_score = score2
                    logger.info(f"Broadened search improved score to {score2:.2f}")

            if not chunks:
                yield sse("token", {"text": (
                    "I could not find relevant information in the programme documents.\n"
                    "Please check the official IITM BS portal: https://study.iitm.ac.in/ds/"
                )})
                yield sse("done", {})
                return

            # 4. Build context
            context_text, sources, images = build_context(chunks)

            # 5. Build messages with history
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            for msg in req.history[-MAX_HISTORY:]:
                messages.append({"role": msg.role, "content": msg.content})

            messages.append({
                "role": "user",
                "content": (
                    f"CONTEXT FROM PROGRAMME DOCUMENTS:\n"
                    f"{context_text}\n\n"
                    f"STUDENT QUESTION: {question}\n\n"
                    f"Instructions:\n"
                    f"- Answer directly and confidently from context above\n"
                    f"- For fee/grading/eligibility questions: include ALL specific numbers, percentages, and details from context — never summarize vaguely\n"
                    f"- If question is yes/no → start with YES or NO\n"
                    f"- Include ALL relevant URLs from context inline where helpful\n"
                    f"- End with source citation — URL is MANDATORY\n"
                    f"- If no URL in context use https://study.iitm.ac.in/ds/\n"
                    f"- Never use 'it seems', 'it appears', 'based on context'\n"
                    f"- Never show internal labels like [SOURCE:...] in your answer"
                )
            })

            # 6. Stream answer
            yield sse("status", {"text": "Generating answer..."})

            stream_resp = llm_call(
                messages   = messages,
                max_tokens = 1200,
                stream     = True,
            )

            for chunk in stream_resp:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield sse("token", {"text": delta.content})
                await asyncio.sleep(0)

            # 7. Send sources + images
            yield sse("sources", {"sources": sources})
            if images:
                yield sse("images", {"images": images})
            yield sse("done", {})

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield sse("error", {"text": str(e)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        }
    )


# ══════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    try:
        info      = qdrant.get_collection(QDRANT_COLLECTION)
        qdrant_ok = f"ok — {info.points_count} points"
    except Exception as e:
        qdrant_ok = f"error: {e}"

    return {
        "status":               "ok",
        "llm_provider":         LLM_PROVIDER,
        "llm_model":            LLM_MODEL,
        "embed_model":          EMBEDDING_MODEL,
        "embed_provider":       "voyage_ai",
        "rerank_model":         RERANKER_MODEL,
        "rerank_provider":      "voyage_ai",
        "qdrant":               qdrant_ok,
        "collection":           QDRANT_COLLECTION,
        "hybrid_search":        f"vector({VECTOR_WEIGHT}) + bm25({BM25_WEIGHT})",
        "retrieval_top_k":      RETRIEVAL_TOP_K_EXPANDED,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)