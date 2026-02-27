"""
vectorstore.py — ChromaDB vector-store management.

Responsibilities
----------------
1. Initialise a *persistent* ChromaDB client (writes to disk).
2. Embed document chunks using HuggingFace sentence-transformers.
3. Create / load the Chroma collection so the 150 PDFs only need to be
   embedded **once**.
4. Expose a LangChain-compatible `retriever` for downstream RAG usage.

Usage
-----
    from vectorstore import build_vectorstore, load_vectorstore

    # First run — embed & persist:
    vectorstore = build_vectorstore(chunks)

    # Subsequent runs — just load from disk:
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    RETRIEVER_TOP_K,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Embedding function (singleton-ish — cached on first call)
# ==============================================================================
_embedding_fn: HuggingFaceEmbeddings | None = None


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Return a HuggingFace embedding function (lazy-loaded, reused).

    Model weights are downloaded on first invocation and cached by
    sentence-transformers under ``~/.cache/``.
    """
    global _embedding_fn
    if _embedding_fn is None:
        logger.info(
            "Loading embedding model '%s' on device '%s' …",
            EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE,
        )
        _embedding_fn = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},  # cosine similarity
        )
        logger.info("Embedding model loaded successfully.")
    return _embedding_fn


# ==============================================================================
# Build (embed + persist) — first-time setup
# ==============================================================================

def build_vectorstore(
    chunks: List[Document],
    persist_directory: Path = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    batch_size: int = 500,
) -> Chroma:
    """
    Embed *chunks* and persist them into a local ChromaDB collection.

    Parameters
    ----------
    chunks : list[Document]
        Pre-chunked LangChain Documents (from ``ingest.py``).
    persist_directory : Path
        Where ChromaDB stores its data on disk.
    collection_name : str
        Name of the Chroma collection.
    batch_size : int
        Number of chunks to embed per batch (keeps memory in check).

    Returns
    -------
    Chroma
        A LangChain Chroma vectorstore instance.
    """
    if not chunks:
        raise ValueError("No chunks provided — nothing to embed.")

    persist_directory.mkdir(parents=True, exist_ok=True)
    embedding_fn = get_embedding_function()

    logger.info(
        "Embedding %d chunks into ChromaDB collection '%s' …",
        len(chunks), collection_name,
    )

    # Process in batches to avoid OOM on large corpora
    vectorstore: Chroma | None = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        logger.info("  Batch %d / %d  (%d chunks)", batch_num, total_batches, len(batch))

        if vectorstore is None:
            # First batch — create the collection
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embedding_fn,
                collection_name=collection_name,
                persist_directory=str(persist_directory),
            )
        else:
            # Subsequent batches — add to existing collection
            vectorstore.add_documents(documents=batch)

    logger.info(
        "✓ Vector store persisted at '%s' (%d vectors).",
        persist_directory, len(chunks),
    )
    return vectorstore  # type: ignore[return-value]


# ==============================================================================
# Load existing persisted store — fast, no re-embedding
# ==============================================================================

def load_vectorstore(
    persist_directory: Path = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """
    Load a previously persisted ChromaDB collection from disk.

    Raises
    ------
    FileNotFoundError
        If the persist directory does not exist.
    """
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at '{persist_directory}'. "
            "Run the ingestion pipeline first (python main.py --ingest)."
        )

    embedding_fn = get_embedding_function()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=str(persist_directory),
    )

    count = vectorstore._collection.count()
    logger.info(
        "Loaded ChromaDB collection '%s' from '%s' — %d vectors.",
        collection_name, persist_directory, count,
    )
    return vectorstore


# ==============================================================================
# Convenience: get a LangChain retriever
# ==============================================================================

def get_retriever(vectorstore: Chroma, top_k: int = RETRIEVER_TOP_K):
    """
    Wrap the Chroma vectorstore as a LangChain retriever.

    Parameters
    ----------
    vectorstore : Chroma
        A loaded or freshly built Chroma vector store.
    top_k : int
        Number of most-similar chunks to retrieve per query.

    Returns
    -------
    langchain_core.retrievers.BaseRetriever
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


# ==============================================================================
# Quick smoke-test
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    vs = load_vectorstore()
    retriever = get_retriever(vs)
    results = retriever.invoke("What is the tuition refund policy?")
    print(f"\n✓ Retrieved {len(results)} chunks.")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.metadata['source']} (chunk {doc.metadata.get('chunk_index', '?')})")
        print(f"      {doc.page_content[:150]}…\n")
