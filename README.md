# Mimic — Conversational AI with Document Intelligence

An end-to-end AI assistant built with FastAPI and Groq's lightning-fast inference. Mimic combines real-time document understanding, hybrid search, live web access, and streaming chat — all through a single WebSocket connection. Powered by Llama 3.3 70B via Groq (free tier), with fully local embeddings.

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/simmy00/AI_Agent.git
cd AI_Agent

# Create virtual environment
python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your GROQ_API_KEY (free from https://console.groq.com)
# Optionally add SERPER_API_KEY for web search

# Launch
uvicorn main:app --reload
# Visit http://localhost:8000
```

> **Optional:** For scanned PDF support, install Tesseract OCR:
> - Linux: `sudo apt install tesseract-ocr poppler-utils`
> - Windows: [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)

---

## How It Works

Mimic follows a **ground-first, reason-second** philosophy. When a user sends a message, the system decides what source to use:

1. **Uploaded documents** — If the user has uploaded files and relevant chunks are found via hybrid search, that context is injected into the prompt with citations.
2. **Live web search** — For queries about current events, news, weather, prices, or anything time-sensitive, Mimic fetches real Google results via Serper and scrapes actual page content.
3. **Calculator** — Math expressions, percentages, and formulas are evaluated through a sandboxed AST parser.
4. **LLM knowledge** — For general questions, explanations, and conversation, the model responds directly.

```
User message
      │
      ▼
┌─────────────────┐     ┌──────────────────┐
│ Intent Detection │────▶│ Tool Execution   │
│ (regex patterns) │     │ (web / calc)     │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│  RAG Context Injection                  │
│  (if documents uploaded + relevant)     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Groq LLM (Llama 3.3 70B)             │
│  Streaming response via WebSocket       │
└─────────────────────────────────────────┘
```

### Why Not `bind_tools`?

I initially used LangChain's `bind_tools()` for structured function calling, but Llama models on Groq produce inconsistent tool call formats — sometimes leaking raw control tokens (`<|end_header_id|>`) instead of proper structured calls. The solution: **pre-LLM intent detection** using pattern matching. Tools execute before the LLM sees the message, and their results are injected as context. This gives 100% reliable tool execution with zero token leakage.

---

## Codebase Overview

```
├── main.py                          # FastAPI app + startup lifecycle
├── requirements.txt
├── .env.example
│
├── frontend/
│   └── index.html                   # Single-file UI (glassmorphism dark theme)
│
├── sessions/                        # Persistent chat history (JSON per session)
├── chroma_db/                       # Vector store (auto-created)
│
└── app/
    ├── api/
    │   └── routes.py                # WebSocket handler — chat, upload, sessions
    │
    ├── agents/
    │   └── rag_agent.py             # Core agent — intent routing, streaming, RAG injection
    │
    ├── core/
    │   ├── config.py                # Pydantic Settings (typed .env loading)
    │   ├── dependencies.py          # Singleton wiring (lru_cache)
    │   ├── session_store.py         # JSON-backed session persistence
    │   └── ws_manager.py            # WebSocket connection registry
    │
    ├── models/
    │   └── schemas.py               # Pydantic models, enums
    │
    ├── prompts/
    │   └── templates.py             # System prompt, RAG template, tool descriptions
    │
    ├── rag/
    │   ├── document_processor.py    # PDF/TXT parsing + OCR fallback + chunking
    │   ├── vector_store.py          # ChromaDB + HuggingFace local embeddings
    │   └── hybrid_retriever.py      # BM25 sparse + dense + RRF fusion
    │
    ├── tools/
    │   └── tools.py                 # Web search (Serper + page scraping) + calculator (safe AST)
    │
    └── validators/
        └── response_validator.py    # Heuristic quality checker (stale-phrase detection)
```

---

## Hybrid Retrieval

Plain embedding search misses exact keyword matches. Plain keyword search misses semantic similarity. Mimic runs both and merges results.

**Dense search** — HuggingFace `all-MiniLM-L6-v2` encodes queries and document chunks into 384-dim vectors. ChromaDB handles cosine similarity.

**Sparse search** — BM25 (in-memory via `rank-bm25`) catches exact terms, IDs, acronyms, and proper nouns that embeddings miss.

**Fusion** — Reciprocal Rank Fusion combines both ranked lists:

```
score(chunk) = 1/(60 + rank_dense) + 1/(60 + rank_sparse)
```

Chunks appearing in both lists score highest. The top-K results are injected into the LLM prompt with source citations.

---

## Real-Time Communication

Everything runs over a single WebSocket at `/ws/main`. No HTTP endpoints for chat or upload — streaming tokens, upload progress, and session management all flow through one persistent connection.

**Key design choice:** `session_id` is sent inside each message body, not in the URL. Switching chat sessions doesn't require disconnecting or reconnecting — the same socket handles everything.

### Message Types

| Direction | Type | Purpose |
|---|---|---|
| Client → Server | `chat` | Send a message (includes `session_id`) |
| Client → Server | `upload` | Send a file as base64 |
| Client → Server | `session_load` | Load a previous session's history |
| Client → Server | `session_delete` | Delete a session |
| Server → Client | `chat_token` | Single streamed token |
| Server → Client | `chat_done` | Response complete |
| Server → Client | `upload_start` | Upload processing began |
| Server → Client | `page_done` | One PDF page processed |
| Server → Client | `upload_complete` | All chunks indexed |
| Server → Client | `searching_web` | Web search in progress |

---

## Web Search Pipeline

When Mimic detects a time-sensitive query, it runs a multi-stage search:

1. **Serper API** — Queries Google via Serper.dev. Extracts answer boxes, knowledge graph cards, and organic results.
2. **Page scraping** — For the top 2 organic results, `httpx` fetches the actual HTML, strips navigation/scripts, and extracts clean text (~2000 chars per page).
3. **Context injection** — All search results (with source URLs) are passed to the LLM, which synthesizes a coherent answer with citations.

---

## Document Processing

1. **Extension validation** — Only `.pdf` and `.txt` accepted (enum-enforced).
2. **Text extraction** — `pypdf` for native PDF text. If a page yields no text, `pytesseract` + `pdf2image` performs OCR.
3. **Chunking** — Text is split into overlapping chunks (default: 600 chars, 100 overlap) to preserve context across boundaries.
4. **Embedding** — Each chunk is encoded locally via HuggingFace `all-MiniLM-L6-v2` (384 dimensions).
5. **Storage** — Chunks and embeddings go into ChromaDB with metadata (filename, page number, chunk ID).
6. **BM25 rebuild** — The in-memory BM25 index is rebuilt after every upload.
7. **Real-time progress** — Page-by-page progress is broadcast to the frontend via WebSocket.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *required* | Free API key from [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model identifier |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformers model |
| `SERPER_API_KEY` | *optional* | Enables live web search |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector database path |
| `SESSIONS_DIR` | `./sessions` | Chat history storage |
| `CHUNK_SIZE` | `600` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `6` | Dense search candidates |
| `BM25_TOP_K` | `6` | Sparse search candidates |
| `HYBRID_FINAL_K` | `4` | Final chunks after fusion |

---

## Technical Choices

**Groq over local Ollama** — Running Llama locally requires 2-8GB disk and a decent GPU. Groq's free tier provides the same Llama 3.3 70B model with sub-second latency and zero hardware requirements.

**Local embeddings over API-based** — Embeddings run in-process via `sentence-transformers`. No network calls, no rate limits, no costs. The `all-MiniLM-L6-v2` model loads once at startup (~90MB) and handles all encoding.

**Pre-LLM tool routing over structured function calling** — Llama models on Groq don't reliably produce structured tool calls through `bind_tools()`. Manual intent detection with regex is deterministic, fast, and never produces leaked tokens or malformed calls.

**Heuristic validator over LLM-based** — A second LLM call per response would double latency. The heuristic checker catches stale-knowledge phrases and short responses without any inference overhead.

**WebSocket over REST** — REST requires polling for streaming. SSE is unidirectional. WebSocket gives full duplex — tokens stream out while heartbeats flow in, upload progress pushes in real time, and session switches happen without reconnection.

**JSON session files over Redis** — Zero infrastructure. Every session is a standalone JSON file. The `session_store.py` interface is abstracted — swapping to Redis or Postgres is a single-file change.

---

## Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| LLM | Llama 3.3 70B via Groq |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local) |
| Vector DB | ChromaDB (embedded, persistent) |
| Sparse search | BM25 via rank-bm25 |
| Web search | Serper.dev + httpx page scraping |
| PDF processing | pypdf + pytesseract + pdf2image |
| Frontend | Vanilla HTML/CSS/JS (single file) |
| Communication | WebSocket (single persistent connection) |
