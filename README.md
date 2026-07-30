# LF Jobs RAG

Hybrid Retrieval-Augmented Generation for natural-language search over a fixed job-listings dataset (`LF_Jobs.xlsx`).

The system retrieves relevant job-description chunks with **dense + sparse** vectors in **Qdrant**, optionally **reranks** them with a cross-encoder, then generates a grounded answer with **Gemini** via **LlamaIndex**. A **FastAPI** backend exposes versioned HTTP endpoints; a thin **Gradio** UI calls that API only.

---

## How it works

1. **Ingestion (already run offline)**  
   Job HTML is cleaned and split with structure-aware chunking. Each chunk is embedded with:
   - **Dense:** `BAAI/bge-base-en-v1.5` (FastEmbed / sentence-transformers path as configured)
   - **Sparse:** `Qdrant/bm25` (FastEmbed), stored with Qdrant’s `Modifier.IDF`  
   Points are upserted into a Qdrant collection (`lf_jobs`) with named vectors `dense` and `sparse`, plus job metadata in the payload.

2. **Query time**  
   - The user sends a natural-language query to **FastAPI** (`GET /api/v1/query`).  
   - On process start, FastAPI **lifespan** loads embedding models, Qdrant client, reranker, and Gemini into a shared **`rag_core.runtime`** (no per-request model load / no reliance on `lru_cache` for production path).  
   - **Retrieval:** hybrid search (RRF over dense + sparse), or dense-/sparse-only.  
   - **Rerank (default on):** cross-encoder over a candidate pool, then keep top‑N.  
   - **Generation:** LlamaIndex `RetrieverQueryEngine` wraps our Qdrant retriever and synthesizes an answer with Gemini, grounded in retrieved excerpts only.

3. **UI**  
   Gradio is a **client**. Submitting a prompt triggers an HTTP request to the FastAPI endpoint; the API runs the full RAG path and returns answer + source chunks.

```
┌─────────┐   HTTP    ┌──────────────┐         ┌────────────────────────────┐
│ Gradio  │ ────────► │ FastAPI app  │ ──────► │ rag_core (retrieve+generate)│
│  (UI)   │           │  /api/v1/*   │         │ Qdrant · rerank · Gemini   │
└─────────┘           └──────────────┘         └────────────────────────────┘
```

---

## Why separate `app/` and `rag_core/` (low coupling)

| Layer | Responsibility | Depends on |
|--------|----------------|------------|
| **`rag_core/`** | Chunking, indexing helpers, retrieval, reranking, generation | Models, Qdrant, LlamaIndex — **no FastAPI, no Gradio** |
| **`app/`** | HTTP API, schemas, SSE orchestration, web config | `rag_core` only at the service boundary |

**Benefits:**

- **Testability:** retrieval and generation can be unit-tested or run from scripts without spinning up a web server.  
- **Replaceability:** swap Gradio, add another client, or change API shape without touching embedding or prompt logic.  
- **Startup clarity:** heavy models are initialized once in the API lifespan and shared via `rag_core.runtime`; the UI never loads models.  
- **Clear ownership:** versioned routes and Pydantic schemas live under `app/`; domain RAG lives under `rag_core/`.

---

## Folder structure

```
app/                              # Web / API layer
├── main.py                       # FastAPI app + lifespan → init_runtime()
├── core/
│   └── config.py                 # API host, port, /api/v1 prefix only
├── schemas/
│   └── query.py                  # QueryRequest / QueryResponse / SourceNode
├── services/
│   └── query_service.py          # Thin orchestrator: sync JSON + SSE → answer_query
└── api/v1/
    ├── router.py                 # Aggregates v1 routers
    └── endpoints/
        └── query.py              # GET /api/v1/query

rag_core/                         # Domain RAG (importable standalone)
├── config.py                     # Models, Qdrant, chunk sizes, Gemini, rerank flags
├── runtime.py                    # Process-wide handles (dense, sparse, Qdrant, LLM, reranker)
├── ingestion/
│   ├── preprocessing.py          # HTML clean + structure-aware chunking
│   ├── create_collection.py      # Qdrant collection: dense + sparse (+ payload indexes)
│   └── index_documents.py        # Embed chunks + upsert into Qdrant
├── retrieval/
│   ├── retriever.py              # Dense / sparse / hybrid (RRF) + optional rerank
│   └── reranker.py               # Cross-encoder over candidate pool
└── generation/
    ├── prompts.py                # Versioned QA / system prompts
    ├── llm.py                    # Gemini via LlamaIndex (uses runtime.llm)
    ├── llama_retriever.py        # Adapter: our Retriever → LlamaIndex BaseRetriever
    └── query_engine.py           # RetrieverQueryEngine + answer_query()

tests/
├── test_generation.py          
├── test_preprocessing.py
└── test_retriever.py              # Smoke-test dense / sparse / hybrid search

app_gradio.py                     # UI client → FastAPI only
data/                             # LF_Jobs.xlsx (not committed if large)
```

### Components by area

**Ingestion**  
Structure-aware chunking (headings, noise/boilerplate filters, size fallbacks), collection creation with named vectors, batch dense+sparse embed and upsert.

**Retrieval**  
Query embedding with the same FastEmbed models as index time; Qdrant search modes; optional cross-encoder rerank using a larger first-stage pool.

**Generation**  
LlamaIndex query engine over the hybrid retriever; Gemini Flash (configurable model id); prompts constrained to retrieved context.

**API**  
Versioned under `/api/v1`; health reports whether runtime models are loaded; entry points for the same underlying `answer_query`.

---

## Getting started

### Prerequisites

- Python 3.11+ recommended  
- [uv](https://github.com/astral-sh/uv) for dependency management  
- **Qdrant** reachable (default `http://localhost:6333`) with the **`lf_jobs` collection already indexed** (see assumptions below)  
- **`GEMINI_API_KEY`** (or `GOOGLE_API_KEY`) for generation  

### Install

```bash
# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
uv sync
```

### Add .env file with following keys:

```
QDRANT_API_KEY=xxx
QDRANT_URL=xxx
GEMINI_API_KEY=xxx
API_BASE_URL=http://127.0.0.1:8000
```

Place `LF_Jobs.xlsx` under `data/` only if you plan to **re-run** ingestion.

### Run the backend

```bash
export PYTHONPATH=$(pwd)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Docs: `http://localhost:8000/docs`  
- Health: `GET /api/v1/health`  
- Query: `POST /api/v1/query`  

### Run the Gradio UI

With the API already running:

```bash
export PYTHONPATH=$(pwd)
# optional if API is not on localhost:8000
# export API_BASE_URL=http://127.0.0.1:8000
python3 app_gradio.py
```

Gradio only calls `GET {API_BASE_URL}/api/v1/query`.

### Optional: rebuild the index

Ingestion is **not** required for normal use if the collection is already built. Re-indexing is slow (thousands of chunks + model downloads):

```bash
uv run rag_core/ingestion/create_collection.py
uv run rag_cpre/ingestion/index_documents.py
```

---

## API surface (v1)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Root status |
| `GET` | `/api/v1/health` | Runtime readiness (models / LLM loaded) |
| `GET` | `/api/v1/query` | Full RAG → string (`answer`) |

Request body (query endpoints):

```json
{
  "query": "marketing internship in New York",
  "top_n": 30,
  "mode": "hybrid",
  "use_reranker": true
}
```

`mode`: `hybrid` | `dense` | `sparse`.

---

## Limitations and assumptions

### Limitations
- **Latency in Generation**: Due to free tier Gemini Key, the generation time can be significantly higher resulting in greater response time. 

- **Token streaming is not enabled.** The SSE endpoint reports coarse stages and then a **full** answer. Gemini token-level streaming is not wired through LlamaIndex synthesis yet.  
- **No conversation history.** Each request is independent.  
- **No query decomposition / multi-hop.** Complex multi-part questions are handled as a single retrieval + generation pass.  
- **Gradio has no file upload.** The corpus is treated as fixed; the UI is search-only.  
- **Evaluation is not automated** in-repo (no retrieval@k / faithfulness suite shipped).

### Assumptions

- **Ingestion is already done.** The job file is stable and consistent, so day-to-day operation does not re-chunk or re-embed. Gradio intentionally omits upload/index flows. You *can* re-run `create_collection` + `index_documents`, but it takes significant time.  
- **Chunking assumptions:** structure-aware splits (pseudo-headings from HTML, boilerplate filtering, size budgets on the order of ~1.2k–2.4k characters with overlap). These heuristics fit this dataset; they are not proven optimal without evaluation.  
- **Candidate pool / top‑k:** the first-stage pool used before reranking is on the order of **~20–40** (config: `RERANK_CANDIDATE_POOL` / prefetch limits). With **5k+ chunks**, that is a pragmatic default, not a measured optimum—**evaluation is required** to choose ideal pool size, final `top_n`, and chunk size.  
- **Qdrant and Gemini** are available at the configured URL/key when the API starts.  
- **Same embedding models at query time as at index time** (dense BGE + `Qdrant/bm25`).

---

## Configuration (high level)

| Variable | Role |
|----------|------|
| `QDRANT_URL` | Qdrant HTTP endpoint |
| `QDRANT_COLLECTION_NAME` / `COLLECTION_NAME` | Collection id (`lf_jobs`) |
| `DENSE_MODEL` | Default `BAAI/bge-base-en-v1.5` |
| `SPARSE_MODEL` | Default `Qdrant/bm25` |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Generation |
| `GEMINI_MODEL` | Default `gemini-3-flash-preview` (override if your account uses another id) |
| `USE_RERANKER` | Default true |
| `RERANK_CANDIDATE_POOL` | First-stage pool before cross-encoder |
| `API_BASE_URL` | Gradio → API base (default `http://127.0.0.1:8000`) |

See `rag_core/config.py` and `app/core/config.py` for the full set.

---

## Future improvements

- **Evaluate retrieval and generation** (e.g. recall@k, nDCG, groundedness / citation accuracy) and tune chunk size, overlap, `top_n`, and rerank pool from data—not defaults.  
- **Stream the response** properly (token-level SSE from Gemini through the query engine).  
- **Query decomposition** for complex or multi-constraint questions (retrieve per sub-query, then fuse).  
- **Chat history** and session memory for follow-ups (“only remote ones”, “compare the top two”).  
- **Metadata filters** (location, level, category) via Qdrant payload filters exposed on the API.  
- **Observability:** latency breakdown (embed / search / rerank / LLM), prompt version tagging in logs.  
- **Optional re-index jobs** (admin-only) if the spreadsheet ever becomes mutable again.

---

## License / data

Job listing content is subject to the rights of the original dataset provider. This project is a search/RAG demonstration architecture over that data, not a redistribution of the spreadsheet itself.