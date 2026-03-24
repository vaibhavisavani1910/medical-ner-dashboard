"""Pydantic models for the RAG pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Document models ──────────────────────────────────────────────────────────

class Document(BaseModel):
    """A text chunk to be embedded and stored in the vector index."""

    id: str = Field(..., description="Unique stable identifier for this document.")
    content: str = Field(..., description="Human-readable text that will be embedded.")
    category: Literal[
        "overview", "conditions", "symptoms", "medications", "procedures", "cooccurrence"
    ]
    metadata: dict = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    """A document returned by vector similarity search."""

    document: Document
    score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity (higher = more relevant).")

    @property
    def snippet(self) -> str:
        """First 200 chars of content, for display in citations."""
        return self.document.content[:200].rstrip() + ("…" if len(self.document.content) > 200 else "")


# ─── Chat models ─────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single turn in the conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    message: str = Field(..., min_length=1, max_length=2000, description="User's question.")
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Prior conversation turns (most recent last).",
    )

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()


class Source(BaseModel):
    """A citation attached to an assistant response."""

    id: str
    category: str
    snippet: str


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """Full response from the RAG pipeline."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    usage: TokenUsage


# ─── Admin models ─────────────────────────────────────────────────────────────

class IndexStatus(BaseModel):
    """State of the vector index."""

    indexed: bool
    document_count: int
    collection_name: str
    message: str
