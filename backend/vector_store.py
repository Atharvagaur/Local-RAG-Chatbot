"""ChromaDB vector store wrapper.

Responsibilities:
- Initialize a persistent ChromaDB client and collection.
- Insert document chunks with embeddings and metadata.
- Prevent duplicate document ingestion using content hashes.
- Perform semantic similarity search with optional document filtering.
- Persist data across application restarts.
- Clear the document database.
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class VectorStore:
    """Wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        embedding: Embeddings,
        persist_directory: Optional[str] = None,
        collection_name: str = "document_chunks",
    ) -> None:
        self.embedding = embedding
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._vectorstore: Optional[Chroma] = None

    def _get_or_create(self) -> Chroma:
        if self._vectorstore is None:
            logger.info(
                "Opening ChromaDB collection '%s'%s",
                self.collection_name,
                f" at {self.persist_directory}" if self.persist_directory else " (in-memory)",
            )
            kwargs: dict = dict(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                client_settings=chromadb.Settings(
                    anonymized_telemetry=False,
                ),
            )
            if self.persist_directory:
                kwargs["persist_directory"] = self.persist_directory
            self._vectorstore = Chroma(**kwargs)
        return self._vectorstore

    def _document_already_indexed(self, document_id: str) -> bool:
        """Check whether a document_id already exists in the collection."""
        try:
            store = self._get_or_create()
            existing = store.get(where={"document_id": document_id}, limit=1)
            return bool(existing and existing.get("ids"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Duplicate check failed for %s: %s", document_id, exc)
            return False

    def add_documents(
        self,
        documents: list[Document],
        document_id: str,
    ) -> int:
        """Add chunks to the vector store.

        Returns the number of chunks added. Skips insertion if the
        document_id is already present (duplicate prevention).
        """
        if not documents:
            return 0

        if self._document_already_indexed(document_id):
            logger.info("Document %s already indexed; skipping", document_id)
            return 0

        store = self._get_or_create()
        store.add_documents(documents)
        logger.info("Added %d chunks for document %s", len(documents), document_id)
        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        document_id: Optional[str] = None,
    ) -> list[Document]:
        """Return the top-k most similar chunks for a query."""
        store = self._get_or_create()
        search_kwargs: dict = {"k": k}
        if document_id:
            search_kwargs["filter"] = {"document_id": document_id}
        return store.similarity_search(query, **search_kwargs)

    def get_collection_count(self) -> int:
        """Return the total number of chunks in the collection."""
        try:
            store = self._get_or_create()
            return store._collection.count()  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to get collection count: %s", exc)
            return 0

    def indexed_document_ids(self) -> list[str]:
        """Return the distinct document_ids currently indexed."""
        try:
            store = self._get_or_create()
            data = store.get()
            ids = data.get("metadatas", [])
            document_ids = {
                m.get("document_id") for m in ids if m and m.get("document_id")
            }
            return sorted(document_ids)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to list indexed documents: %s", exc)
            return []

    def clear(self) -> None:
        """Delete the entire collection."""
        try:
            store = self._get_or_create()
            store.delete_collection()
            logger.info("Cleared ChromaDB collection '%s'", self.collection_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Clear collection failed: %s", exc)
        finally:
            self._vectorstore = None
