# AUA Policy Bot

A local **Retrieval-Augmented Generation (RAG)** system that answers questions about American University of Armenia (AUA) policies using your own PDF documents — entirely offline, no API keys required.

## Architecture

```
PDF Documents → PyMuPDF → Text Chunks → SentenceTransformers → ChromaDB
                                                                    ↓
                  User Question → Embed → Similarity Search → Top-K Chunks
                                                                    ↓
                                                    Prompt + Context → Ollama/Llama-3 → Answer
```

**Stack:**
| Component | Library |
|-----------|---------|
| PDF parsing | PyMuPDF |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU) |
| Vector store | ChromaDB (persistent, local) |
| LLM | Ollama + Llama 3 (runs locally) |
| CLI | argparse + Rich |

## Prerequisites

- **Python 3.10+**
- **Ollama** — <https://ollama.com/download>

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/GurgenHovakimyan/aua_policy_bot.git
cd aua_policy_bot

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install & start Ollama, then pull the model
ollama pull llama3
```

## Quick Start

### 1. Add your PDF documents

Place your PDF files in the `documents/` folder:

```
aua_policy_bot/
├── documents/
│   ├── Grading_Policy.pdf
│   ├── Smoke_Free_Environment.pdf
│   └── ...
```

### 2. Ingest documents

```bash
python main.py --ingest
```

This loads all PDFs, splits them into chunks, embeds them, and persists the vector store to `chroma_db/`.

### 3. Ask questions

**Single query:**
```bash
python main.py --query "Can I smoke at AUA?"
```

**Interactive mode:**
```bash
python main.py --interactive
```

**Both modes support `--top-k` to control how many chunks are retrieved:**
```bash
python main.py -q "What is the grading policy?" -k 3
```

## Configuration

All settings are in [`config.py`](config.py) and can be overridden via environment variables or a `.env` file (see [`.env.example`](.env.example)).

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Characters per text chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu`, `cuda`, or `mps` |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `LLM_TEMPERATURE` | 0.1 | LLM sampling temperature |
| `LLM_MAX_TOKENS` | 256 | Max output tokens |
| `RETRIEVER_TOP_K` | 5 | Number of chunks to retrieve |

## Project Structure

```
aua_policy_bot/
├── config.py          # Central configuration (paths, models, prompts)
├── ingest.py          # PDF loading & text chunking
├── vectorstore.py     # ChromaDB embedding, persistence & retrieval
├── rag_chain.py       # RAG chain construction & query helpers
├── main.py            # CLI entry-point (--ingest, --query, --interactive)
├── requirements.txt   # Pinned Python dependencies
├── .env.example       # Template for environment variable overrides
└── documents/         # Place your PDF files here (not tracked by git)
```

## Usage Examples

```
$ python main.py --interactive

============================================================
  AUA Policy RAG — Interactive Mode
============================================================
Loading embedding model ... OK
Opening ChromaDB ... OK (831 vectors)
Connecting to LLM ... OK

Ready! Ask your question.

You: Can I smoke at AUA?
  [1/3] Embedding query ... OK
  [2/3] Searching documents ... OK (5 chunks)
  [3/3] Generating answer ... OK