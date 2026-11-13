"""Ollama embedding initialization and configuration.

Responsibilities:
- Initialize OllamaEmbeddings with the configured embedding model.
- Provide a cached, reusable embedding instance.
- Validate that the embedding model exists in Ollama.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages the Ollama embedding model instance."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        ollama_manager=None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self._ollama_manager = ollama_manager
        self._embeddings: Optional[OllamaEmbeddings] = None

    def validate(self) -> None:
        """Ensure the embedding model is installed locally."""
        if self._ollama_manager is not None:
            self._ollama_manager.validate_model(
                self.model_name, purpose="embedding"
            )

    def get_embeddings(self) -> OllamaEmbeddings:
        """Return a reusable OllamaEmbeddings instance."""
        if self._embeddings is None:
            logger.info("Initializing OllamaEmbeddings(model=%s)", self.model_name)
            self._embeddings = OllamaEmbeddings(
                model=self.model_name,
                base_url=self.base_url,
            )
        return self._embeddings

    @property
    def embeddings(self) -> OllamaEmbeddings:
        return self.get_embeddings()


def build_embedding_manager(
    model_name: str,
    base_url: str,
    ollama_manager=None,
) -> EmbeddingManager:
    """Factory for EmbeddingManager."""
    return EmbeddingManager(
        model_name=model_name,
        base_url=base_url,
        ollama_manager=ollama_manager,
    )
