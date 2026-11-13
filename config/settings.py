"""Centralized application configuration.

Loads environment variables from a ``.env`` file (if present) and exposes
typed configuration values used across the application.
"""

from __future__ import annotations

import os
from typing import Final

from dotenv import load_dotenv

load_dotenv()


def _get_env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


# Application
APP_TITLE: Final[str] = _get_env("APP_TITLE", "Document Intelligence AI")
APP_DESCRIPTION: Final[str] = _get_env(
    "APP_DESCRIPTION",
    "Local RAG chatbot for PDF and TXT documents using Ollama, LangChain and ChromaDB",
)

# Ollama
OLLAMA_BASE_URL: Final[str] = _get_env("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBEDDING_MODEL: Final[str] = _get_env(
    "DEFAULT_EMBEDDING_MODEL", "mxbai-embed-large:latest"
)
DEFAULT_GENERATION_MODEL: Final[str] = _get_env(
    "DEFAULT_GENERATION_MODEL", "qwen3:4b"
)
EMBEDDING_MODEL: Final[str] = _get_env("EMBEDDING_MODEL", "mxbai-embed-large:latest")

# Chunking
CHUNK_SIZE: Final[int] = _get_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP: Final[int] = _get_int("CHUNK_OVERLAP", 200)

# Retrieval
DEFAULT_TOP_K: Final[int] = _get_int("DEFAULT_TOP_K", 4)
MAX_TOP_K: Final[int] = _get_int("MAX_TOP_K", 10)

# Conversation memory
MAX_CONVERSATION_MESSAGES: Final[int] = _get_int(
    "MAX_CONVERSATION_MESSAGES", 10
)

# Vector store
CHROMA_PERSIST_DIRECTORY: Final[str] = _get_env(
    "CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"
)
CHROMA_COLLECTION_NAME: Final[str] = _get_env(
    "CHROMA_COLLECTION_NAME", "document_chunks"
)

# Logging
LOG_LEVEL: Final[str] = _get_env("LOG_LEVEL", "INFO")

# Supported file extensions
SUPPORTED_EXTENSIONS: Final[tuple[str, ...]] = (".pdf", ".txt")
