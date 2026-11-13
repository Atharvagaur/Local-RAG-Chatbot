"""Tests for document_processor.py.

These tests do not require a running Ollama server.
"""

from __future__ import annotations

import pytest

from backend.document_processor import (
    compute_file_hash,
    load_document,
    process_document,
    split_text,
    DocumentProcessorError,
)


def test_compute_file_hash_is_stable():
    content = b"hello world"
    assert compute_file_hash(content) == compute_file_hash(content)
    assert compute_file_hash(b"different") != compute_file_hash(content)


def test_load_txt():
    content = b"Line one.\nLine two."
    text, num_pages, page_texts = load_document(content, "sample.txt")
    assert num_pages == 1
    assert "Line one." in text
    assert len(page_texts) == 1


def test_load_unsupported_format_raises():
    with pytest.raises(DocumentProcessorError):
        load_document(b"data", "sample.docx")


def test_process_txt_creates_chunks_with_metadata():
    content = b"Sentence one. " * 200  # large enough to split
    processed = process_document(
        content, "sample.txt", chunk_size=100, chunk_overlap=20
    )
    assert processed.error is None
    assert processed.num_chunks > 1
    for chunk in processed.chunks:
        assert chunk.metadata["source"] == "sample.txt"
        assert "document_id" in chunk.metadata
        assert "chunk_id" in chunk.metadata
        assert "page" in chunk.metadata
        assert chunk.page_content.strip()


def test_process_empty_text_returns_error():
    processed = process_document(b"", "empty.txt")
    assert processed.error is not None
    assert processed.num_chunks == 0


def test_process_whitespace_only_text_returns_error():
    processed = process_document(b"   \n   ", "blank.txt")
    assert processed.error is not None


def test_duplicate_detection_by_hash():
    content = b"Unique content for hashing."
    h1 = compute_file_hash(content)
    h2 = compute_file_hash(content)
    assert h1 == h2
    different = b"Different content."
    assert compute_file_hash(different) != h1


def test_split_text_preserves_metadata():
    page_texts = [("Hello world. " * 50, 1), ("Second page content. " * 50, 2)]
    docs = split_text(page_texts, "file.pdf", "doc-123", chunk_size=80, chunk_overlap=10)
    pages = {d.metadata["page"] for d in docs}
    assert pages == {1, 2}
    for d in docs:
        assert d.metadata["source"] == "file.pdf"
        assert d.metadata["document_id"] == "doc-123"
        assert d.metadata["chunk_id"].startswith("doc-123_")


def test_process_pdf_extracts_text_and_metadata(monkeypatch):
    from backend import document_processor

    class FakePage:
        def extract_text(self):
            return "This is a PDF document about artificial intelligence."

    class FakeReader:
        def __init__(self, stream):
            self.pages = [FakePage()]

    monkeypatch.setattr(document_processor, "PdfReader", FakeReader)

    processed = process_document(b"%PDF-1.4 fake pdf bytes", "ai.pdf")
    assert processed.error is None, processed.error
    assert processed.num_pages == 1
    assert processed.num_chunks >= 1
    full_text = " ".join(c.page_content for c in processed.chunks)
    assert "artificial intelligence" in full_text
    assert processed.chunks[0].metadata["source"] == "ai.pdf"
    assert processed.chunks[0].metadata["page"] == 1


def test_process_corrupted_pdf_returns_error():
    processed = process_document(b"%PDF-1.4 corrupted garbage not a real pdf", "bad.pdf")
    assert processed.error is not None
    assert processed.num_chunks == 0
