"""
config.py — Central configuration for the RAG pipeline.

All tuneable hyper-parameters live here so every other module stays clean.
Override any value via a .env file or environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env in project root, if present

# Disable ChromaDB telemetry (avoids posthog compatibility warnings)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

# ==============================================================================
# Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"   # persisted vector store

# ==============================================================================
# PDF Ingestion
# ==============================================================================
SUPPORTED_EXTENSIONS = {".pdf"}

# ==============================================================================
# Text Splitting  (RecursiveCharacterTextSplitter)
# ------------------------------------------------------------------------------
# chunk_size   = 1000 chars  → ~200-250 tokens (good for MiniLM 256-token window)
# chunk_overlap = 200 chars  → preserves context across chunk boundaries
# ==============================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# ==============================================================================
# Embedding Model  (HuggingFace Sentence-Transformers)
# ==============================================================================
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
# Device for embedding inference: "cpu", "cuda", "mps"
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# ==============================================================================
# ChromaDB
# ==============================================================================
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "aua_policies")

# ==============================================================================
# LLM  (Ollama — local Llama-3)
# ------------------------------------------------------------------------------
# Make sure Ollama is running:  ollama serve
# And the model is pulled:      ollama pull llama3
# ==============================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 256))  # max output tokens

# ==============================================================================
# Retrieval
# ==============================================================================
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", 5))  # number of chunks to fetch

# ==============================================================================
# Prompt Template
# ==============================================================================
RAG_PROMPT_TEMPLATE = """You are an expert assistant for the American University of Armenia (AUA).
Use ONLY the following context extracted from AUA policy documents to answer the question.
If the answer is not contained in the context, say "I could not find the answer in the provided documents."
Be concise — answer in 2-4 sentences maximum. Cite the source document name.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}

Answer (concise, 2-4 sentences):"""
