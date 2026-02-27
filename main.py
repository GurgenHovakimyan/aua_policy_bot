"""
main.py — CLI entry-point for the AUA Policy RAG System.

Commands
--------
    python main.py --ingest       Load PDFs, chunk, embed, and persist to ChromaDB.
    python main.py --query "…"    Ask a question against the persisted vector store.
    python main.py --interactive  Start an interactive Q&A loop in the terminal.

All three modes can be combined:
    python main.py --ingest --interactive
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import CHROMA_PERSIST_DIR, RETRIEVER_TOP_K

console = Console()

# ==============================================================================
# Logging setup
# ==============================================================================

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy third-party loggers
    for name in ("httpx", "httpcore", "chromadb", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ==============================================================================
# Ingestion pipeline
# ==============================================================================

def run_ingestion() -> None:
    """Load PDFs → chunk → embed → persist to ChromaDB."""
    from ingest import load_and_chunk_documents
    from vectorstore import build_vectorstore

    console.print(Panel("[bold cyan]Starting document ingestion pipeline[/bold cyan]"))
    start = time.time()

    # Step 1: Load & chunk
    with console.status("[bold green]Loading and chunking PDFs…"):
        chunks = load_and_chunk_documents()
    console.print(f"  [green]✓[/green] {len(chunks)} chunks created.\n")

    # Step 2: Embed & persist
    with console.status("[bold green]Embedding chunks into ChromaDB…"):
        build_vectorstore(chunks)

    elapsed = time.time() - start
    console.print(
        f"\n  [green]✓[/green] Ingestion complete in {elapsed:.1f}s.  "
        f"ChromaDB persisted at [bold]{CHROMA_PERSIST_DIR}[/bold]\n"
    )


# ==============================================================================
# Single query
# ==============================================================================

def run_single_query(question: str, top_k: int) -> None:
    """Answer one question and print the result."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_ollama import ChatOllama
    from config import (
        CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
        EMBEDDING_MODEL_NAME, RAG_PROMPT_TEMPLATE,
        LLM_MAX_TOKENS, LLM_TEMPERATURE, OLLAMA_BASE_URL, OLLAMA_MODEL,
    )
    from rag_chain import _format_docs

    console.print(f"\n[bold yellow]Question:[/bold yellow] {question}\n")

    # Step 1: Embed query
    print("  [1/3] Loading model & embedding query ...", end=" ", flush=True)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = embed_model.encode(question).tolist()
    print("OK", flush=True)

    # Step 2: Query ChromaDB
    print("  [2/3] Searching documents ...", end=" ", flush=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = client.get_collection(CHROMA_COLLECTION_NAME)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    retrieved_docs = []
    for doc_text, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved_docs.append(Document(page_content=doc_text, metadata=meta or {}))
    print(f"OK ({len(retrieved_docs)} chunks)", flush=True)

    # Step 3: Generate answer
    context = _format_docs(retrieved_docs)
    filled_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    ).format(context=context, question=question)

    print("  [3/3] Generating answer ...", end=" ", flush=True)
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_predict=LLM_MAX_TOKENS,
    )
    answer_msg = llm.invoke(filled_prompt)
    answer_text = StrOutputParser().invoke(answer_msg)
    print("OK", flush=True)

    # Answer
    console.print(Panel(answer_text, title="Answer", border_style="green"))

    # Sources table
    if retrieved_docs:
        table = Table(title="Retrieved Sources", show_lines=True)
        table.add_column("#", style="bold", width=3)
        table.add_column("Document", style="cyan")
        table.add_column("Chunk", justify="center")
        table.add_column("Preview", max_width=80)
        for i, doc in enumerate(retrieved_docs, 1):
            table.add_row(
                str(i),
                doc.metadata.get("source", "?"),
                str(doc.metadata.get("chunk_index", "?")),
                doc.page_content[:120].replace("\n", " ") + "…",
            )
        console.print(table)


# ==============================================================================
# Interactive loop
# ==============================================================================

def run_interactive(top_k: int) -> None:
    """Start a REPL-style Q&A session."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_ollama import ChatOllama
    from config import (
        CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
        EMBEDDING_MODEL_NAME, RAG_PROMPT_TEMPLATE,
        LLM_MAX_TOKENS, LLM_TEMPERATURE, OLLAMA_BASE_URL, OLLAMA_MODEL,
    )
    from rag_chain import _format_docs

    print("=" * 60, flush=True)
    print("  AUA Policy RAG — Interactive Mode", flush=True)
    print("  Type your question and press Enter.", flush=True)
    print("  Type 'quit' or 'exit' to stop.", flush=True)
    print("=" * 60, flush=True)

    # Load embedding model directly (no LangChain wrapper)
    print("Loading embedding model ...", end=" ", flush=True)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("OK", flush=True)

    # Open ChromaDB directly (no LangChain wrapper)
    print("Opening ChromaDB ...", end=" ", flush=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = client.get_collection(CHROMA_COLLECTION_NAME)
    print(f"OK ({collection.count()} vectors)", flush=True)

    # Connect to Ollama
    print("Connecting to LLM ...", end=" ", flush=True)
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_predict=LLM_MAX_TOKENS,
    )
    print("OK", flush=True)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )
    parser = StrOutputParser()
    print("\nReady! Ask your question.\n", flush=True)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        try:
            # Step 1: Embed query directly with sentence-transformers
            print("  [1/3] Embedding query ...", end=" ", flush=True)
            query_embedding = embed_model.encode(question).tolist()
            print("OK", flush=True)

            # Step 2: Query ChromaDB directly
            print("  [2/3] Searching documents ...", end=" ", flush=True)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas"],
            )
            # Convert to LangChain Documents for _format_docs
            retrieved_docs = []
            for doc_text, meta in zip(results["documents"][0], results["metadatas"][0]):
                retrieved_docs.append(Document(page_content=doc_text, metadata=meta or {}))
            print(f"OK ({len(retrieved_docs)} chunks)", flush=True)

            # Step 3: Format context and call LLM
            context = _format_docs(retrieved_docs)
            filled_prompt = prompt.format(context=context, question=question)

            print("  [3/3] Generating answer ...", end=" ", flush=True)
            answer_msg = llm.invoke(filled_prompt)
            answer_text = parser.invoke(answer_msg)
            print("OK", flush=True)

            # Print answer
            print(f"\nAssistant: {answer_text}\n", flush=True)
        except Exception as exc:
            print(f"\nError: {exc}", flush=True)
            import traceback
            traceback.print_exc()


# ==============================================================================
# Argument parser
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AUA Policy RAG System — Local Retrieval-Augmented Generation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run the ingestion pipeline (load PDFs → embed → persist).",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help='Ask a single question, e.g.  --query "What is the grading policy?"',
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive Q&A session.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=RETRIEVER_TOP_K,
        help=f"Number of chunks to retrieve (default: {RETRIEVER_TOP_K}).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ==============================================================================
# Entry-point
# ==============================================================================

def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    if not args.ingest and not args.query and not args.interactive:
        console.print(
            "[bold red]No action specified.[/bold red]  "
            "Use --ingest, --query, or --interactive.\n"
            "Run  [bold]python main.py --help[/bold]  for details."
        )
        sys.exit(1)

    if args.ingest:
        run_ingestion()

    if args.query:
        run_single_query(args.query, top_k=args.top_k)

    if args.interactive:
        run_interactive(top_k=args.top_k)


if __name__ == "__main__":
    main()
