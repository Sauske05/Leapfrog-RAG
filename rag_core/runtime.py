"""
Process-wide runtime handles (models, clients).

Populated once at FastAPI startup via init_runtime().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastembed import SparseTextEmbedding, TextEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from rag_core.config import (
    DENSE_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_TEMPERATURE,
    QDRANT_API_KEY,
    QDRANT_URL,
    RERANKER_MODEL_NAME,
    SPARSE_MODEL,
)


@dataclass
class Runtime:
    dense_model: Optional[TextEmbedding] = None
    sparse_model: Optional[SparseTextEmbedding] = None
    llm: Optional[GoogleGenAI] = None
    qdrant: Optional[QdrantClient] = None
    reranker: Optional[CrossEncoder] = None
    ready: bool = False


runtime = Runtime()


def init_runtime(*, load_reranker: bool = True, load_llm: bool = True) -> Runtime:
    """Load models/clients once. Safe to call again (no-op if already ready)."""
    if runtime.ready:
        return runtime

    print("Runtime: loading dense embedding model...")
    runtime.dense_model = TextEmbedding(model_name=DENSE_MODEL)

    print("Runtime: loading sparse BM25 model...")
    runtime.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    print("Runtime: connecting to Qdrant...")
    runtime.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if load_reranker:
        print("Runtime: loading cross-encoder reranker...")
        runtime.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    if load_llm:
        if not GEMINI_API_KEY:
            print("Runtime: GEMINI_API_KEY missing — LLM not loaded.")
        else:
            print(f"Runtime: loading Gemini LLM ({GEMINI_MODEL})...")
            runtime.llm = GoogleGenAI(
                model=GEMINI_MODEL,
                api_key=GEMINI_API_KEY,
                temperature=LLM_TEMPERATURE,
            )

    runtime.ready = True
    print("Runtime: ready.")
    return runtime


def get_runtime() -> Runtime:
    if not runtime.ready:
        raise RuntimeError(
            "Runtime not initialized. Call init_runtime() at app startup."
        )
    return runtime