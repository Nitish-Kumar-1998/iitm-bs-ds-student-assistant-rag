"""
IITM BS RAG Pipeline — Central Configuration
=============================================
Single source of truth for entire pipeline.
Switch between public (Groq) and private (Ollama)
with one flag: USE_LOCAL_LLM = True/False

Public mode  (USE_LOCAL_LLM = False) → Groq API
Private mode (USE_LOCAL_LLM = True)  → Ollama local
"""
import os
from pathlib import Path
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
# ══════════════════════════════════════════════════════════════════
# PRIVACY MODE — flip this one flag to switch entire system
# ══════════════════════════════════════════════════════════════════

USE_LOCAL_LLM = False  # True = Ollama (private), False = Groq (public)

# ══════════════════════════════════════════════════════════════════
# LLM CONFIGURATION
# ══════════════════════════════════════════════════════════════════

if USE_LOCAL_LLM:
    # ── Ollama (private mode) ──────────────────────────────────
    LLM_BASE_URL    = "http://localhost:11434/v1"
    LLM_API_KEY     = "ollama"                    # dummy key, required by openai client
    LLM_MODEL       = "llama3.1"                  # change to llama3.1:70b if you have GPU
    LLM_PROVIDER    = "ollama"
else:
    # ── Groq (public mode) ────────────────────────────────────
    LLM_BASE_URL    = "https://api.groq.com/openai/v1"
    LLM_API_KEY     = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL       = "llama-3.3-70b-versatile"
    LLM_PROVIDER    = "groq"

# LLM fallback models (Groq only)
LLM_FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "gemma2-9b-it",
] if not USE_LOCAL_LLM else []


# ══════════════════════════════════════════════════════════════════
# EMBEDDING & RERANKER CONFIGURATION
# Voyage AI API — no local models, zero RAM overhead
# ══════════════════════════════════════════════════════════════════

VOYAGE_API_KEY      = os.getenv("VOYAGE_API_KEY", "")
EMBEDDING_MODEL     = "voyage-3"                  # state of art 2024
EMBEDDING_DIM       = 1024                        # voyage-3 output dim
EMBEDDING_BATCH     = 128                         # voyage supports larger batches
RERANKER_MODEL      = "rerank-2"                  # voyage reranker
RERANKER_TOP_K      = 5


# ══════════════════════════════════════════════════════════════════
# VECTOR DB CONFIGURATION
# ══════════════════════════════════════════════════════════════════


QDRANT_HOST         = "localhost"
QDRANT_PORT         = 6333
QDRANT_COLLECTION   = "iitm_bs"

QDRANT_URL          = os.getenv("QDRANT_URL", f"http://{QDRANT_HOST}:{QDRANT_PORT}")
QDRANT_API_KEY      = os.getenv("QDRANT_API_KEY", None)

# Hybrid search weights
VECTOR_WEIGHT       = 0.7                         # semantic search weight
BM25_WEIGHT         = 0.3                         # keyword search weight

# Retrieval settings
RETRIEVAL_TOP_K     = 20                          # fetch top 20 before reranking
RERANK_TOP_K        = 4                           # keep top 5 after reranking

# ══════════════════════════════════════════════════════════════════
# SCRAPER CONFIGURATION
# ══════════════════════════════════════════════════════════════════

ROOT_DOCS = [
    {
        "url": (
            "https://docs.google.com/document/d/e/"
            "2PACX-1vSUvKzH7yIXNVwUgRYSIT8M0x1jhFSkslEtj9UPo3dtWI_sJ38Hh_"
            "PzbBygpF0vIOo8K7lTy-uYkqdu/pub"
        ),
        "title": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "filename": "programme_guide.md",
    },
    {
        "url": (
            "https://docs.google.com/document/d/e/"
            "2PACX-1vRxGnnDCVAO3KX2CGtMIcJQuDrAasVk2JHbDxkjsGrTP5ShhZK8N6ZSPX89lex"
            "Kx86QPAUswSzGLsOA/pub"
        ),
        "title": "Student Handbook",
        "filename": "student_handbook.md",
    },
]

# Only follow Google Doc links
FOLLOW_URL_PATTERN  = r"https://docs\.google\.com/document/"

# Skip these domains during scraping
SKIP_DOMAINS = [
    "drive.google.com",
    "support.google.com",
    "accounts.google.com",
]

REQUEST_TIMEOUT     = 15                          # seconds per request
DELAY_BETWEEN       = 1.0                         # seconds between requests

# ══════════════════════════════════════════════════════════════════
# CHUNKER CONFIGURATION
# ══════════════════════════════════════════════════════════════════

CHUNK_SIZE          = 512                         # max tokens per chunk
CHUNK_OVERLAP       = 50                          # token overlap between chunks
MIN_CHUNK_SIZE      = 50                          # minimum tokens, merge if smaller

# Chunk types
CHUNK_TYPES = [
    "text",                                       # regular paragraph content
    "table",                                      # markdown table content
    "image",                                      # OCR'd image content
    "reference_link",                             # external/sheet link metadata
    "restricted_doc",                             # blocked doc metadata
    "section_index",                              # section navigation index
]

# ══════════════════════════════════════════════════════════════════
# HYDE CONFIGURATION
# ══════════════════════════════════════════════════════════════════

HYDE_QUESTIONS_PER_CHUNK = 3                      # questions to generate per chunk
HYDE_SKIP_TYPES = [                               # don't generate HyDE for these
    "section_index",
    "restricted_doc",
]

HYDE_PROMPT = """You are helping build a RAG system for IITM BS (IIT Madras BS Degree) students.

Given this content from the IITM BS programme documents:

DOCUMENT: {doc_title}
SECTION: {section}
CONTENT: {content}

Generate exactly {n} questions that an IITM BS student would ask 
whose answer is found in this content.

Rules:
- Questions must be answerable ONLY from this content
- ALWAYS include the specific course name, level, or topic in the question
- If content is about a specific course, ALWAYS name that course in every question
- If content is about fees, ALWAYS specify which level (foundation/diploma/degree)
- If content is about exams, ALWAYS specify which exam and which course
- Use natural student language (casual, direct)
- Cover different aspects of the content
- Do not repeat similar questions
- Return ONLY the questions, one per line, no numbering"""
# ══════════════════════════════════════════════════════════════════
# IMAGE SCANNER CONFIGURATION
# ══════════════════════════════════════════════════════════════════

IMAGE_SCAN_ALL      = True                        # scan every image, no skipping
IMAGE_MIN_SIZE_KB   = 1                           # still scan but flag if very small

IMAGE_PROMPT = """This image is from an IITM BS (IIT Madras BS Degree) programme document.
Section: {section}
Context before image: {before}
Context after image: {after}

Analyze this image and:
1. If it contains a TABLE: extract ALL data from the table in markdown format
2. If it contains a DIAGRAM or FLOWCHART: describe the structure and all labels clearly
3. If it contains TEXT: extract the text exactly
4. If it is DECORATIVE (logo, banner, illustration with no data): respond with exactly: DECORATIVE

Be thorough and extract every piece of information visible."""

# ══════════════════════════════════════════════════════════════════
# PATHS — all output locations
# ══════════════════════════════════════════════════════════════════

BASE_DIR            = Path(__file__).parent
OUTPUT_DIR          = BASE_DIR / "output"
DOCS_DIR            = OUTPUT_DIR / "docs"
LINKED_DIR          = DOCS_DIR / "linked_docs"
IMAGES_DIR          = OUTPUT_DIR / "images"
CHUNKS_DIR          = OUTPUT_DIR / "chunks"

# Output files
CHECKPOINT_FILE     = OUTPUT_DIR / "checkpoint.json"
SKIPPED_LOG         = OUTPUT_DIR / "skipped_links.log"
IMAGE_METADATA_FILE = OUTPUT_DIR / "image_metadata.json"
REFERENCE_LINKS_FILE= OUTPUT_DIR / "reference_links.json"
RESTRICTED_LINKS_FILE= OUTPUT_DIR / "restricted_links.json"
DOC_REGISTRY_FILE   = OUTPUT_DIR / "document_registry.json"
ALL_CHUNKS_FILE     = CHUNKS_DIR / "all_chunks.json"
EVAL_SET_FILE       = BASE_DIR / "eval_set.json"
EVAL_REPORT_FILE    = BASE_DIR / "eval_report.json"

# ══════════════════════════════════════════════════════════════════
# RAG CONFIGURATION
# ══════════════════════════════════════════════════════════════════

RAG_SYSTEM_PROMPT = """You are a helpful assistant for IITM BS (IIT Madras BS Degree in Data Science) students.

Answer questions accurately based ONLY on the provided context.
Always cite your sources.

Rules:
1. Answer directly and clearly
2. If the answer has a specific number, date or policy — state it exactly
3. Always end with: Source: [document name, section name]
4. If a relevant link exists in context — always include it
5. If you cannot find the answer in context — say exactly:
   "I could not find this information in the IITM BS documents.
    Please check: https://study.iitm.ac.in/ds/"
6. Never make up information
7. If answer is in an external sheet or link — provide that link directly"""

# RAG streaming
STREAM_RESPONSE     = True

# ══════════════════════════════════════════════════════════════════
# SCHEDULER CONFIGURATION
# ══════════════════════════════════════════════════════════════════

SCRAPE_INTERVAL_WEEKS = 13                        # every 3 months

# ══════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════

LOG_LEVEL           = "INFO"
LOG_FORMAT          = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# ══════════════════════════════════════════════════════════════════
# VALIDATION — catch missing config on startup
# ══════════════════════════════════════════════════════════════════

def validate():
    """Call this at startup of any pipeline stage."""
    errors = []

    if not USE_LOCAL_LLM and not LLM_API_KEY:
        errors.append("GROQ_API_KEY environment variable not set")

    if not VOYAGE_API_KEY:
        errors.append("VOYAGE_API_KEY environment variable not set")

    if not QDRANT_URL or QDRANT_URL == f"http://{QDRANT_HOST}:{QDRANT_PORT}":
        import socket
        try:
            socket.connect_ex(("localhost", QDRANT_PORT))
        except Exception:
            errors.append(f"Qdrant not reachable at {QDRANT_HOST}:{QDRANT_PORT}")

    if errors:
        print("\n❌ Configuration errors:")
        for e in errors:
            print(f"   → {e}")
        raise SystemExit(1)

    print(f"\n✅ Config loaded:")
    print(f"   LLM:        {LLM_PROVIDER} → {LLM_MODEL}")
    print(f"   Embeddings: {EMBEDDING_MODEL} (Voyage AI)")
    print(f"   Reranker:   {RERANKER_MODEL} (Voyage AI)")
    print(f"   Vector DB:  {QDRANT_URL}/{QDRANT_COLLECTION}")
    print(f"   Mode:       {'🔒 Private (local)' if USE_LOCAL_LLM else '🌐 Public (API)'}")


if __name__ == "__main__":
    validate()

