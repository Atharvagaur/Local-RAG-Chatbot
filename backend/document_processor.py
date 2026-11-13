"""Document loading, validation, and chunking.

Responsibilities:
- Load PDF and TXT files from in-memory bytes.
- Extract text and page-level metadata.
- Split text into overlapping chunks using RecursiveCharacterTextSplitter.
- Generate stable content hashes for duplicate detection.
- Attach metadata (source, page, document id, chunk id) to every chunk.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class DocumentProcessorError(ValueError):
    """Raised for document loading / processing failures."""


@dataclass
class ProcessedDocument:
    """Result of processing a single uploaded file."""

    filename: str
    file_hash: str
    num_pages: int
    num_chunks: int
    chunks: list[Document] = field(default_factory=list)
    error: Optional[str] = None


def compute_file_hash(content: bytes) -> str:
    """Return a stable SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


def _extract_pdf(content: bytes, filename: str) -> tuple[str, int, list[tuple[str, int]]]:
    """Extract text from a PDF.

    Returns (full_text, num_pages, list_of_(text, page_number)).
    """
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise DocumentProcessorError(
            f"Failed to read PDF '{filename}': {exc}"
        ) from exc

    pages_text: list[tuple[str, int]] = []
    num_pages = len(reader.pages)

    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to extract page %d of %s: %s", index, filename, exc)
            text = ""
        if text.strip():
            pages_text.append((text, index))

    full_text = "\n".join(text for text, _ in pages_text)
    return full_text, num_pages, pages_text


def _extract_txt(content: bytes, filename: str) -> tuple[str, int, list[tuple[str, int]]]:
    """Extract text from a TXT file."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    return text, 1, [(text, 1)] if text.strip() else []


def load_document(content: bytes, filename: str) -> tuple[str, int, list[tuple[str, int]]]:
    """Load a document and return (text, num_pages, page_texts)."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf(content, filename)
    if lower.endswith(".txt"):
        return _extract_txt(content, filename)
    raise DocumentProcessorError(
        f"Unsupported file type: {filename}. Only PDF and TXT are supported."
    )


def split_text(
    page_texts: list[tuple[str, int]],
    filename: str,
    document_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split page texts into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    documents: list[Document] = []
    chunk_index = 0

    for text, page_number in page_texts:
        chunks = splitter.split_text(text)
        for chunk_text in chunks:
            if not chunk_text.strip():
                continue
            metadata = {
                "source": filename,
                "page": page_number,
                "document_id": document_id,
                "chunk_id": f"{document_id}_{chunk_index}",
                "chunk_index": chunk_index,
            }
            documents.append(
                Document(page_content=chunk_text, metadata=metadata)
            )
            chunk_index += 1

    return documents


def process_document(
    content: bytes,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> ProcessedDocument:
    """Process one uploaded file into a ProcessedDocument."""
    file_hash = compute_file_hash(content)
    document_id = str(uuid.uuid4())

    try:
        full_text, num_pages, page_texts = load_document(content, filename)
    except DocumentProcessorError as exc:
        logger.error("Failed to load %s: %s", filename, exc)
        return ProcessedDocument(
            filename=filename,
            file_hash=file_hash,
            num_pages=0,
            num_chunks=0,
            error=str(exc),
        )

    if not full_text or not full_text.strip():
        logger.warning("No extractable text in %s", filename)
        return ProcessedDocument(
            filename=filename,
            file_hash=file_hash,
            num_pages=num_pages,
            num_chunks=0,
            error="No extractable text found in the document.",
        )

    chunks = split_text(page_texts, filename, document_id, chunk_size, chunk_overlap)

    logger.info(
        "Processed %s: %d pages, %d chunks (hash=%s)",
        filename,
        num_pages,
        len(chunks),
        file_hash,
    )

    return ProcessedDocument(
        filename=filename,
        file_hash=file_hash,
        num_pages=num_pages,
        num_chunks=len(chunks),
        chunks=chunks,
    )
