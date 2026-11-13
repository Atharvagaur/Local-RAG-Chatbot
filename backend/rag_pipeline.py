"""RAG question-answering pipeline.

Responsibilities:
- Retrieve relevant chunks from the vector store.
- Build a context-aware prompt with strict grounding instructions.
- Pass context + question to the selected Ollama LLM.
- Attribute sources to retrieved chunks.
- Support question contextualization for follow-ups.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant answering questions based on uploaded documents.

GROUND RULES:
1. Prioritize information from the retrieved document context.
2. Do not invent facts or fabricate information.
3. If the answer is not available in the retrieved context, explicitly say so.
4. Clearly distinguish information from the uploaded documents from your general knowledge.
5. Provide detailed explanations when necessary, but avoid unnecessarily long answers.
6. Use the retrieved context rather than relying entirely on pretrained knowledge.

When the context is sufficient, answer using the documents. When it is not, say you cannot answer from the provided documents.
"""

RETRIEVAL_PROMPT_TEMPLATE = """You are helping to convert a follow-up question into a standalone question for document retrieval.

Conversation history:
{conversation}

Follow-up question: {question}

Rewrite the follow-up question as a complete, standalone question that can be used to search the documents. Do not answer it. Only output the rewritten question.
"""

QA_PROMPT_TEMPLATE = """{system_prompt}

Retrieved document context:
{context}

Conversation history (for context only):
{conversation}

Question: {question}

Answer:"""


@dataclass
class RagResult:
    """Result of a RAG query."""

    answer: str
    sources: list[dict]
    context_chunks: list[Document]
    latency_seconds: float
    model_used: str


class RagPipeline:
    """End-to-end RAG pipeline."""

    def __init__(
        self,
        vector_store,
        chat_model: BaseChatModel,
        chat_manager=None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        top_k: int = 4,
    ) -> None:
        self.vector_store = vector_store
        self.chat_model = chat_model
        self.chat_manager = chat_manager
        self.system_prompt = system_prompt
        self.top_k = top_k
        self._output_parser = StrOutputParser()

    def _format_context(self, chunks: list[Document]) -> str:
        if not chunks:
            return "No relevant document context found."
        blocks = []
        for index, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "?")
            blocks.append(
                f"[Source {index}] Document: {source} | Page: {page}\n{chunk.page_content}"
            )
        return "\n\n".join(blocks)

    def _extract_sources(self, chunks: list[Document]) -> list[dict]:
        sources = []
        for chunk in chunks:
            sources.append(
                {
                    "filename": chunk.metadata.get("source", "unknown"),
                    "page": chunk.metadata.get("page", "?"),
                    "snippet": chunk.page_content[:500],
                    "document_id": chunk.metadata.get("document_id", ""),
                    "chunk_id": chunk.metadata.get("chunk_id", ""),
                }
            )
        return sources

    def _contextualize(self, question: str) -> str:
        if self.chat_manager is None:
            return question
        contextualized = self.chat_manager.contextualize_question(question)
        if contextualized != question:
            try:
                prompt = ChatPromptTemplate.from_template(RETRIEVAL_PROMPT_TEMPLATE)
                chain = prompt | self.chat_model | self._output_parser
                standalone = chain.invoke(
                    {
                        "conversation": self.chat_manager.get_conversation_string(),
                        "question": question,
                    }
                )
                if standalone and standalone.strip():
                    return standalone.strip()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Contextualization failed: %s", exc)
        return contextualized

    def retrieve(self, question: str) -> list[Document]:
        """Retrieve top-k relevant chunks for a (possibly contextualized) question."""
        return self.vector_store.similarity_search(question, k=self.top_k)

    def generate(
        self,
        question: str,
        chunks: list[Document],
        conversation: str = "",
    ) -> RagResult:
        """Generate an answer using retrieved context and return sources."""
        start = time.perf_counter()

        context = self._format_context(chunks)
        sources = self._extract_sources(chunks)

        prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)
        chain = prompt | self.chat_model | self._output_parser

        try:
            answer = chain.invoke(
                {
                    "system_prompt": self.system_prompt,
                    "context": context,
                    "conversation": conversation or "No previous conversation.",
                    "question": question,
                }
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise RuntimeError(
                "Failed to generate an answer. The Ollama model may be unavailable."
            ) from exc

        latency = time.perf_counter() - start
        model_name = getattr(self.chat_model, "model", "unknown")

        return RagResult(
            answer=answer if isinstance(answer, str) else str(answer),
            sources=sources,
            context_chunks=chunks,
            latency_seconds=round(latency, 3),
            model_used=model_name,
        )

    def query(
        self,
        question: str,
        use_contextualization: bool = True,
    ) -> RagResult:
        """Run the full RAG pipeline for a user question."""
        if not question or not question.strip():
            raise ValueError("Please enter a non-empty question.")

        retrieval_question = self._contextualize(question) if use_contextualization else question
        chunks = self.retrieve(retrieval_question)

        conversation = ""
        if self.chat_manager is not None:
            conversation = self.chat_manager.get_conversation_string()

        result = self.generate(question, chunks, conversation=conversation)

        # Preserve the original question for display/history
        if self.chat_manager is not None:
            self.chat_manager.add_user_message(question)
            self.chat_manager.add_ai_message(result.answer, sources=result.sources)

        return result

    def compare_models(
        self,
        question: str,
        model_a: BaseChatModel,
        model_b: BaseChatModel,
    ) -> tuple[RagResult, RagResult]:
        """Run the same question+context through two models sequentially."""
        retrieval_question = self._contextualize(question)
        chunks = self.retrieve(retrieval_question)
        context = self._format_context(chunks)
        sources = self._extract_sources(chunks)
        conversation = (
            self.chat_manager.get_conversation_string()
            if self.chat_manager
            else ""
        )

        prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)
        parser = StrOutputParser()

        def run(model: BaseChatModel) -> RagResult:
            chain = prompt | model | parser
            start = time.perf_counter()
            try:
                answer = chain.invoke(
                    {
                        "system_prompt": self.system_prompt,
                        "context": context,
                        "conversation": conversation or "No previous conversation.",
                        "question": question,
                    }
                )
            except Exception as exc:
                logger.error("Generation failed for %s: %s", model, exc)
                raise RuntimeError(
                    f"Generation failed for model {getattr(model, 'model', 'unknown')}"
                ) from exc
            latency = round(time.perf_counter() - start, 3)
            return RagResult(
                answer=answer if isinstance(answer, str) else str(answer),
                sources=sources,
                context_chunks=chunks,
                latency_seconds=latency,
                model_used=getattr(model, "model", "unknown"),
            )

        result_a = run(model_a)
        result_b = run(model_b)
        return result_a, result_b
