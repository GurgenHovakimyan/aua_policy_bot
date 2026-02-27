"""
rag_chain.py — Retrieval-Augmented Generation chain.

Responsibilities
----------------
1. Build a LangChain RAG chain that couples:
   • ChromaDB retriever  (from vectorstore.py)
   • Ollama Llama-3 LLM  (local, no API keys needed)
   • A grounded prompt template that constrains the LLM to the retrieved context.
2. Expose a simple ``query()`` function for the main script / notebook.

Usage
-----
    from rag_chain import build_rag_chain

    chain = build_rag_chain()          # uses persisted ChromaDB
    answer = chain.invoke("What is the tuition refund policy?")
    print(answer)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from config import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RAG_PROMPT_TEMPLATE,
    RETRIEVER_TOP_K,
)
from vectorstore import get_retriever, load_vectorstore

logger = logging.getLogger(__name__)


# ==============================================================================
# LLM initialisation
# ==============================================================================

def get_llm() -> ChatOllama:
    """
    Return a ChatOllama client pointed at the locally running Ollama server.

    Prerequisites
    -------------
    1. Install Ollama:  https://ollama.com/download
    2. Pull the model:  ``ollama pull llama3``
    3. Start the server: ``ollama serve``   (runs on port 11434 by default)
    """
    logger.info(
        "Connecting to Ollama LLM  model='%s'  url='%s'  temp=%.2f",
        OLLAMA_MODEL, OLLAMA_BASE_URL, LLM_TEMPERATURE,
    )
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_predict=LLM_MAX_TOKENS,  # cap output length
    )


# ==============================================================================
# Helpers
# ==============================================================================

def _format_docs(docs: List[Document]) -> str:
    """
    Concatenate retrieved document chunks into a single context string.
    Each chunk is prefixed with its source file name for traceability.
    """
    formatted_parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        header = f"[Source {i}: {source} | chunk {chunk_idx}]"
        formatted_parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted_parts)


# ==============================================================================
# RAG chain construction
# ==============================================================================

def build_rag_chain(top_k: int = RETRIEVER_TOP_K):
    """
    Assemble the full RAG pipeline:

        query  →  retriever  →  prompt + context  →  LLM  →  answer string

    Returns
    -------
    A LangChain Runnable that accepts a ``str`` query and returns a ``str``
    answer.
    """
    # 1. Load persisted vector store & wrap as retriever
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore, top_k=top_k)

    # 2. Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )

    # 3. LLM
    llm = get_llm()

    # 4. Chain (LCEL)
    #    The retriever feeds "context"; the original query feeds "question".
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain assembled (top_k=%d).", top_k)
    return chain


# ==============================================================================
# High-level query function  (returns answer + source docs)
# ==============================================================================

def query(
    question: str,
    top_k: int = RETRIEVER_TOP_K,
    return_sources: bool = True,
) -> Dict[str, Any]:
    """
    One-shot convenience function: retrieves context, generates an answer,
    and optionally returns the source document chunks.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    top_k : int
        Number of chunks to retrieve.
    return_sources : bool
        If True, also return the raw retrieved Document objects.

    Returns
    -------
    dict
        ``{"answer": str, "sources": list[Document] | None}``
    """
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore, top_k=top_k)

    # Retrieve
    retrieved_docs = retriever.invoke(question)
    context = _format_docs(retrieved_docs)

    # Generate
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )
    llm = get_llm()

    filled_prompt = prompt.format(context=context, question=question)
    answer = llm.invoke(filled_prompt)
    answer_text = StrOutputParser().invoke(answer)

    return {
        "answer": answer_text,
        "sources": retrieved_docs if return_sources else None,
    }


# ==============================================================================
# Smoke-test
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    chain = build_rag_chain()
    q = "What is the policy on academic freedom at AUA?"
    print(f"\nQuery: {q}\n")
    result = chain.invoke(q)
    print(f"Answer:\n{result}")
