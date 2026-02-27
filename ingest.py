"""
ingest.py — Document ingestion & chunking module.

Responsibilities
----------------
1. Scan the ./documents/ directory for PDF files.
2. Extract native text from each PDF using PyMuPDF (fitz) — no OCR.
3. Split extracted text into overlapping chunks using LangChain's
   RecursiveCharacterTextSplitter.
4. Return a list of LangChain Document objects ready for embedding.

Usage
-----
    from ingest import load_and_chunk_documents
    docs = load_and_chunk_documents()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 1.  PDF text extraction
# ==============================================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Open a single PDF with PyMuPDF and concatenate text from every page.

    Parameters
    ----------
    pdf_path : Path
        Absolute or relative path to the PDF file.

    Returns
    -------
    str
        The full raw text of the document.

    Raises
    ------
    RuntimeError
        If PyMuPDF cannot open or read the file.
    """
    try:
        doc = fitz.open(str(pdf_path))
        pages_text: list[str] = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")  # native text extraction
            if text.strip():
                pages_text.append(text)
        doc.close()
        full_text = "\n\n".join(pages_text)
        logger.debug(
            "Extracted %d chars from %s (%d pages)",
            len(full_text), pdf_path.name, page_num,
        )
        return full_text
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF '{pdf_path}': {exc}") from exc


# ==============================================================================
# 2.  Load all PDFs → LangChain Documents
# ==============================================================================

def load_documents(directory: Path = DOCUMENTS_DIR) -> List[Document]:
    """
    Iterate over every PDF in *directory* and convert each to a
    LangChain `Document` with metadata ``{"source": "<filename>"}``.

    Parameters
    ----------
    directory : Path
        Folder containing PDF files.

    Returns
    -------
    list[Document]
        One Document per PDF (full text, not yet chunked).
    """
    if not directory.exists():
        raise FileNotFoundError(f"Documents directory not found: {directory}")

    pdf_paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {directory}. "
            "Ensure the folder contains .pdf files."
        )

    logger.info("Found %d PDF file(s) in '%s'.", len(pdf_paths), directory)

    documents: list[Document] = []
    skipped = 0

    for pdf_path in pdf_paths:
        try:
            text = extract_text_from_pdf(pdf_path)
            if not text.strip():
                logger.warning("Empty text extracted from '%s' — skipping.", pdf_path.name)
                skipped += 1
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_path.name},
                )
            )
        except RuntimeError as exc:
            logger.error(str(exc))
            skipped += 1

    logger.info(
        "Successfully loaded %d document(s) (%d skipped).",
        len(documents), skipped,
    )
    return documents


# ==============================================================================
# 3.  Chunk documents
# ==============================================================================

def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller, overlapping chunks.

    Uses ``RecursiveCharacterTextSplitter`` which tries to split on
    paragraphs → sentences → words, keeping semantic units intact.

    Parameters
    ----------
    documents : list[Document]
        Full-text documents returned by :func:`load_documents`.
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Overlap between consecutive chunks (preserves context).

    Returns
    -------
    list[Document]
        Chunked documents, each retaining the original metadata plus a
        ``chunk_index`` key.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Add a per-document chunk index to metadata for traceability
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        idx = source_counters.get(src, 0)
        chunk.metadata["chunk_index"] = idx
        source_counters[src] = idx + 1

    logger.info(
        "Split %d document(s) into %d chunk(s) "
        "(chunk_size=%d, overlap=%d).",
        len(documents), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


# ==============================================================================
# 4.  Convenience wrapper
# ==============================================================================

def load_and_chunk_documents(
    directory: Path = DOCUMENTS_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    End-to-end: load all PDFs from *directory*, chunk them, and return
    a flat list of LangChain Document objects ready for embedding.
    """
    raw_docs = load_documents(directory)
    return chunk_documents(raw_docs, chunk_size, chunk_overlap)


# ==============================================================================
# Quick smoke-test when run directly
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    chunks = load_and_chunk_documents()
    print(f"\n✓ Total chunks: {len(chunks)}")
    if chunks:
        sample = chunks[0]
        print(f"  Sample chunk source : {sample.metadata['source']}")
        print(f"  Sample chunk length : {len(sample.page_content)} chars")
        print(f"  Preview: {sample.page_content[:200]}…")
