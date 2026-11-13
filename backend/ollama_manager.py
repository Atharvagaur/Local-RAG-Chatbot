"""Ollama connectivity and model management.

Responsibilities:
- Check that the Ollama service is reachable.
- Discover installed models via the Ollama REST API.
- Separate chat (generation) models from embedding models.
- Exclude cloud models from local-only selections.
- Validate that a given model is available locally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or a required model is missing."""


@dataclass(frozen=True)
class OllamaModel:
    """Metadata for a single installed Ollama model."""

    name: str
    digest: str
    size: int
    modified_at: str

    @property
    def is_cloud_model(self) -> bool:
        """Heuristic: cloud models carry a ':...cloud...' model tag."""
        tag = self.name.rsplit(":", 1)[-1] if ":" in self.name else self.name
        return "cloud" in tag.lower()


@dataclass(frozen=True)
class ModelCatalog:
    """Categorized view of installed Ollama models."""

    chat_models: list[OllamaModel]
    embedding_models: list[OllamaModel]
    all_models: list[OllamaModel]


class OllamaManager:
    """Thin wrapper around the Ollama REST API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def is_available(self, timeout: float = 3.0) -> bool:
        """Return True if the Ollama service responds to /api/tags."""
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags", timeout=timeout
            )
            return response.status_code == 200
        except Exception as exc:  # connectivity may raise OSError/RequestException
            logger.debug("Ollama availability check failed: %s", exc)
            return False

    def list_models(self) -> list[OllamaModel]:
        """Fetch installed models from Ollama.

        Raises OllamaError if the service is unreachable.
        """
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to list Ollama models: %s", exc)
            raise OllamaError(
                "Ollama service is unavailable. Make sure Ollama is running."
            ) from exc

        data = response.json()
        models: list[OllamaModel] = []
        for entry in data.get("models", []):
            models.append(
                OllamaModel(
                    name=entry.get("name", ""),
                    digest=entry.get("digest", ""),
                    size=int(entry.get("size", 0) or 0),
                    modified_at=entry.get("modified_at", ""),
                )
            )
        return models

    def get_catalog(self) -> ModelCatalog:
        """Return installed models split into chat and embedding groups."""
        models = self.list_models()
        chat_models = [m for m in models if not m.is_cloud_model]
        embedding_models = [
            m for m in models if "embed" in m.name.lower()
        ]
        return ModelCatalog(
            chat_models=chat_models,
            embedding_models=embedding_models,
            all_models=models,
        )

    def model_exists(self, model_name: str) -> bool:
        """Return True if a model with the given name is installed."""
        try:
            return any(m.name == model_name for m in self.list_models())
        except OllamaError:
            return False

    def validate_model(self, model_name: str, purpose: str = "generation") -> None:
        """Raise OllamaError if the model is not installed locally."""
        if not self.model_exists(model_name):
            raise OllamaError(
                f"Model '{model_name}' is not installed locally. "
                f"Pull it with: ollama pull {model_name}"
            )


def build_ollama_manager(base_url: str) -> OllamaManager:
    """Factory for OllamaManager."""
    return OllamaManager(base_url)
