"""
config.py
---------
Single source of truth for all API stage settings.

No logic lives here — only constants and environment variables.
Every other file in this package imports from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


# =============================================================================
# SERVER
# =============================================================================

APP_TITLE   = "IITM BS RAG API"
APP_VERSION = "2.0.0"

HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))

# CORS — open for now, matches the v1 deployment. Tighten later by
# replacing ["*"] with the specific Vercel frontend URL once stable.
CORS_ALLOW_ORIGINS = ["*"]


# =============================================================================
# CONVERSATION HISTORY
# =============================================================================
# Number of most recent history messages forwarded to the LLM as context.
# Matches the v1 value — enough for follow-up question resolution without
# blowing up the prompt size on long conversations.

MAX_HISTORY_MESSAGES = 6


# =============================================================================
# OFF-TOPIC DETECTION
# =============================================================================
# Lightweight keyword check to short-circuit obviously off-topic or
# conversational messages (greetings, small talk) before spending a
# retrieval + generation round-trip on them.

OFF_TOPIC_PATTERNS = [
    "who are you", "what are you", "what can you do", "what do you know",
    "how do you work", "are you an ai", "are you a bot",
    "hello", "hi ", "hey ", "thanks", "thank you", "bye", "goodbye",
    "what is the capital", "who is the president", "weather",
    "recipe", "movie", "song", "game",
]

OFF_TOPIC_RESPONSE = (
    "I'm the IITM BS Programme Assistant. I can help you with:\n\n"
    "- Courses and registration\n"
    "- Fees and payments\n"
    "- Exams (quizzes, OPPE, end-term)\n"
    "- Projects (MLP, BDM, App Dev)\n"
    "- Credit transfer (NPTEL, SWAYAM)\n"
    "- Degree requirements and pathways\n\n"
    "What would you like to know about the IITM BS programme?"
)


# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


# =============================================================================
# VALIDATION
# =============================================================================

def validate() -> None:
    """
    Validate that downstream stages (retrieve, generate) are properly
    configured. Delegates to their own validate() functions rather than
    duplicating checks — this stage only adds its own concerns on top.
    """
    print("\n✅  API config validated")
    print(f"    Title:   {APP_TITLE} v{APP_VERSION}")
    print(f"    Host:    {HOST}:{PORT}")
    print(f"    CORS:    {CORS_ALLOW_ORIGINS}")

    # Delegate to retrieve and generate stage validation
    from app.retrieve.config import validate as validate_retrieve
    from app.generate.config import validate as validate_generate

    validate_retrieve()
    validate_generate()


if __name__ == "__main__":
    validate()
