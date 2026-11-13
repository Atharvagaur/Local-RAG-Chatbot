"""Conversation history and follow-up question handling.

Responsibilities:
- Maintain chat history in a structured format.
- Contextualize follow-up questions into standalone queries.
- Trim history to a bounded number of recent messages.
- Preserve the original user question for display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatMessage:
    """A single message in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    sources: list[dict] = field(default_factory=list)


class ChatManager:
    """Manages conversation history and question contextualization."""

    def __init__(self, max_messages: int = 10) -> None:
        self.max_messages = max_messages
        self.history: list[ChatMessage] = []

    def add_user_message(self, content: str) -> ChatMessage:
        message = ChatMessage(role="user", content=content)
        self.history.append(message)
        self._trim()
        return message

    def add_ai_message(
        self, content: str, sources: Optional[list[dict]] = None
    ) -> ChatMessage:
        message = ChatMessage(
            role="assistant", content=content, sources=sources or []
        )
        self.history.append(message)
        self._trim()
        return message

    def get_recent_messages(self, limit: Optional[int] = None) -> list[ChatMessage]:
        """Return the most recent messages, bounded by max_messages."""
        limit = limit or self.max_messages
        return self.history[-limit:]

    def get_conversation_string(self, limit: Optional[int] = None) -> str:
        """Format recent history as a transcript for the LLM."""
        messages = self.get_recent_messages(limit)
        lines = []
        for message in messages:
            role = "User" if message.role == "user" else "Assistant"
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines)

    def contextualize_question(self, question: str) -> str:
        """Convert a follow-up question into a standalone query.

        This is a rule-based contextualization that prepends relevant
        conversation context when the question appears to be a follow-up
        (short, uses pronouns, or references prior turns).
        """
        recent = self.get_recent_messages(limit=6)
        if len(recent) <= 1:
            return question

        lower = question.strip()
        # Tokenize crudely (strip common punctuation) so we match whole words
        # rather than substrings (e.g. "it" inside "capital").
        tokens = {w.strip(".,!?:;\"'()[]") for w in lower.split()}
        single_word_indicators = {
            "that", "it", "this", "those", "these", "its", "their",
            "previous", "earlier", "further", "more",
            "compare", "advantages", "disadvantages", "explain",
        }
        phrase_indicators = ("the above",)
        is_followup = (
            bool(tokens & single_word_indicators)
            or any(phrase in lower for phrase in phrase_indicators)
            or (len(tokens) <= 2 and len(recent) > 1)
        )

        if not is_followup:
            return question

        context_parts = []
        for message in recent[-4:-1]:  # exclude current user turn
            role = "User" if message.role == "user" else "Assistant"
            context_parts.append(f"{role}: {message.content}")
        context = " | ".join(context_parts)

        return f"Context from previous conversation: {context}\n\nCurrent question: {question}"

    def clear(self) -> None:
        self.history = []

    def _trim(self) -> None:
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages :]
