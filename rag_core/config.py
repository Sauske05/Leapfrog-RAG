"""
Pipeline config for ingestion + retrieval.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "LF_Jobs.xlsx"

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "lf_jobs")

#Gemini
GEMINI_API_KEY=os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL='gemini-3-flash-preview'
LLM_TEMPERATURE=0.1

# --- Dense (BGE) ---
DENSE_MODEL = os.getenv("DENSE_MODEL", "BAAI/bge-base-en-v1.5")
DENSE_DIM = 768
DENSE_VECTOR_NAME = "dense"

# --- Sparse (BM25 via FastEmbed) ---
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
SPARSE_VECTOR_NAME = "sparse"

# --- Indexing ---
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
PREFETCH_LIMIT = int(os.getenv("PREFETCH_LIMIT", "40"))

# --- Chunking (used by preprocessing) ---
CHUNK_TARGET_CHARS = 1800
CHUNK_MAX_CHARS = 2400
CHUNK_MIN_CHARS = 120
CHUNK_OVERLAP_CHARS = 200
SEMANTIC_EMBED_MODEL = DENSE_MODEL

# --- Retrieval / generation (kept for later stages) ---
DEFAULT_TOP_N = 40
VECTOR_WEIGHT = 0.6
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Reranker in retrieval 
USE_RERANKER = True
RERANK_CANDIDATE_POOL=30
