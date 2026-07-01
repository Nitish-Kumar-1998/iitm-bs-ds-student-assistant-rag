"""
schemas.py
----------
Pydantic request/response models for the API.

Kept separate from app.py so the data contract is easy to find, version,
and reuse (e.g. if a future admin endpoint needs the same ChatMessage shape).
"""

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """
    A single turn in the conversation history sent by the frontend.
    role is "user" or "assistant", matching OpenAI/Groq message format.
    """
    role: str
    content: str


class AskRequest(BaseModel):
    """
    Request body for POST /ask.

    Matches the v1 API contract exactly — question + optional history —
    so the existing frontend works against this backend with zero changes.
    """
    question: str
    history: list[ChatMessage] = []
