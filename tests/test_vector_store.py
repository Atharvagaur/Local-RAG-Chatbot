"""Tests for vector_store.py.

Uses a fake embedding function so no Ollama server is required.
Each test uses a unique temporary persistence directory; directories are
best-effort cleaned up because ChromaDB may hold a file lock on Windows.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.vector_store import VectorStore


class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings for offline tests."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t)) / 100.0, 0.1, 0.2] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(len(text)) / 100.0, 0.1, 0.2]


def _make_chunks(document_id: str, count: int = 3) -> List[Document]:
    docs = []
    for i in range(count):
        docs.append(
            Document(
                page_content=f"Content chunk {i} about apples and oranges.",
                metadata={
                    "source": f"doc_{document_id}.txt",
                    "page": i + 1,
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_{i}",
                },
            )
        )
    return docs


def _new_store(collection_name: str):
    tmpdir = tempfile.mkdtemp(prefix="chroma_test_")
    vs = VectorStore(
        embedding=FakeEmbeddings(),
        persist_directory=tmpdir,
        collection_name=collection_name,
    )
    return vs, tmpdir


def _cleanup(vs, tmpdir) -> None:
    del vs
    gc.collect()
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_add_and_similarity_search():
    vs, tmpdir = _new_store("test_col_search")
    try:
        chunks = _make_chunks("doc1")
        added = vs.add_documents(chunks, "doc1")
        assert added == 3
        assert vs.get_collection_count() == 3

        results = vs.similarity_search("apples", k=2)
        assert len(results) == 2
        assert all("apples" in r.page_content for r in results)
    finally:
        _cleanup(vs, tmpdir)


def test_source_metadata_retrieval():
    vs, tmpdir = _new_store("test_col_meta")
    try:
        chunks = _make_chunks("doc2")
        vs.add_documents(chunks, "doc2")
        results = vs.similarity_search("apples", k=1)
        assert results[0].metadata["source"] == "doc_doc2.txt"
        assert results[0].metadata["page"] in (1, 2, 3)
        assert results[0].metadata["document_id"] == "doc2"
    finally:
        _cleanup(vs, tmpdir)


def test_duplicate_prevention():
    vs, tmpdir = _new_store("test_col_dup")
    try:
        chunks = _make_chunks("doc3")
        first = vs.add_documents(chunks, "doc3")
        second = vs.add_documents(chunks, "doc3")
        assert first == 3
        assert second == 0
        assert vs.get_collection_count() == 3
    finally:
        _cleanup(vs, tmpdir)


def test_clear_database():
    vs, tmpdir = _new_store("test_col_clear")
    try:
        vs.add_documents(_make_chunks("doc4"), "doc4")
        assert vs.get_collection_count() == 3
        vs.clear()
        assert vs.get_collection_count() == 0
    finally:
        _cleanup(vs, tmpdir)


def test_indexed_document_ids():
    vs, tmpdir = _new_store("test_col_ids")
    try:
        vs.add_documents(_make_chunks("docA"), "docA")
        vs.add_documents(_make_chunks("docB"), "docB")
        ids = set(vs.indexed_document_ids())
        assert ids == {"docA", "docB"}
    finally:
        _cleanup(vs, tmpdir)
