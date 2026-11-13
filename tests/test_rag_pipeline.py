"""Tests for rag_pipeline.py.

Uses a fake chat model and fake vector store so no Ollama server is required.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

import pytest

from backend.chat_manager import ChatManager
from backend.rag_pipeline import DEFAULT_SYSTEM_PROMPT, RagPipeline


class FakeChatModel(Runnable):
    """A deterministic fake chat model capturing the prompt input as text."""

    model = "fake"

    def __init__(self, response: str = "Fake generated answer.") -> None:
        self.response = response
        self.last_input = ""

    def invoke(self, input, config=None, **kwargs):  # type: ignore[no-untyped-def]
        if hasattr(input, "to_messages"):
            parts = [getattr(m, "content", "") for m in input.to_messages()]
            self.last_input = "\n".join(parts)
        elif isinstance(input, dict):
            self.last_input = str(input)
        else:
            self.last_input = str(input)
        return AIMessage(content=self.response)


class FakeVectorStore:
    """Fake vector store returning fixed chunks and capturing the query."""

    def __init__(self, chunks: list[Document]) -> None:
        self.chunks = chunks
        self.last_query: Optional[str] = None
        self.last_k: Optional[int] = None

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> list[Document]:
        self.last_query = query
        self.last_k = k
        return self.chunks[:k]


def _make_chunks() -> list[Document]:
    return [
        Document(
            page_content="Apples are red fruits grown in temperate climates.",
            metadata={
                "source": "fruits.pdf",
                "page": 1,
                "document_id": "doc-1",
                "chunk_id": "doc-1_0",
            },
        ),
        Document(
            page_content="Oranges are citrus fruits rich in vitamin C.",
            metadata={
                "source": "fruits.pdf",
                "page": 2,
                "document_id": "doc-1",
                "chunk_id": "doc-1_1",
            },
        ),
    ]


def test_rag_prompt_contains_context_and_question():
    chunks = _make_chunks()
    fake_model = FakeChatModel()
    vs = FakeVectorStore(chunks)
    pipeline = RagPipeline(vs, fake_model, top_k=4)
    result = pipeline.query("What color are apples?")

    assert result.answer == "Fake generated answer."
    assert result.latency_seconds >= 0
    assert result.model_used == "fake"

    prompt_input = fake_model.last_input
    assert "Apples are red fruits" in prompt_input
    assert "What color are apples?" in prompt_input
    assert DEFAULT_SYSTEM_PROMPT in prompt_input
    assert vs.last_query == "What color are apples?"


def test_source_attribution_matches_chunks():
    chunks = _make_chunks()
    fake_model = FakeChatModel()
    vs = FakeVectorStore(chunks)
    pipeline = RagPipeline(vs, fake_model, top_k=4)
    result = pipeline.query("Question")

    assert len(result.sources) == 2
    assert result.sources[0]["filename"] == "fruits.pdf"
    assert result.sources[0]["page"] == 1
    assert result.sources[1]["filename"] == "fruits.pdf"
    assert result.sources[1]["page"] == 2


def test_empty_question_raises():
    chunks = _make_chunks()
    fake_model = FakeChatModel()
    vs = FakeVectorStore(chunks)
    pipeline = RagPipeline(vs, fake_model, top_k=4)
    with pytest.raises(ValueError):
        pipeline.query("")
    with pytest.raises(ValueError):
        pipeline.query("   ")


def test_conversation_history_passed_to_model():
    chunks = _make_chunks()
    fake_model = FakeChatModel()
    vs = FakeVectorStore(chunks)
    chat_manager = ChatManager()
    chat_manager.add_user_message("Previous question about apples.")
    chat_manager.add_ai_message("Apples are red.")
    pipeline = RagPipeline(vs, fake_model, chat_manager, top_k=4)
    pipeline.query("Tell me more.")

    assert "Previous question about apples." in fake_model.last_input
    assert "Apples are red." in fake_model.last_input


def test_contextualization_retrieves_with_standalone_question():
    chunks = _make_chunks()
    fake_model = FakeChatModel()
    vs = FakeVectorStore(chunks)
    chat_manager = ChatManager()
    chat_manager.add_user_message("Tell me about apples.")
    chat_manager.add_ai_message("Apples are red fruits.")
    pipeline = RagPipeline(vs, fake_model, chat_manager, top_k=4)
    pipeline.query("What are its advantages?")

    # Retrieval should have run with a (contextualized) query.
    assert vs.last_query is not None
    assert len(vs.last_query) > 0
    # Conversation history should be included in the generation prompt.
    assert "Apples are red fruits" in fake_model.last_input


def test_top_k_respected():
    chunks = _make_chunks()
    fake_model = FakeChatModel()
    vs = FakeVectorStore(chunks)
    pipeline = RagPipeline(vs, fake_model, top_k=1)
    pipeline.query("Question")
    assert vs.last_k == 1
