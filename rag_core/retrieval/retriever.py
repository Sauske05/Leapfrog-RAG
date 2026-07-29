"""
Hybrid retriever over Qdrant (dense BGE + sparse BM25 via FastEmbed).

Modes:
  - dense   : cosine over the "dense" named vector
  - sparse  : BM25 over the "sparse" named vector
  - hybrid  : Qdrant RRF fusion of both (default)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)

from rag_core.config import (
    COLLECTION_NAME,
    DEFAULT_TOP_N,
    DENSE_MODEL,
    DENSE_VECTOR_NAME,
    PREFETCH_LIMIT,
    QDRANT_API_KEY,
    QDRANT_URL,
    RERANK_CANDIDATE_POOL,
    SPARSE_MODEL,
    SPARSE_VECTOR_NAME,
    USE_RERANKER,
)
from rag_core.retrieval.reranker import rerank

SearchMode = Literal["dense", "sparse", "hybrid"]


@lru_cache(maxsize=1)
def _get_dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=DENSE_MODEL)


@lru_cache(maxsize=1)
def _get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_MODEL)


def _get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _embed_query(query: str) -> tuple[list[float], SparseVector]:
    dense_vec = list(_get_dense_model().embed([query]))[0].tolist()
    sparse_raw = list(_get_sparse_model().embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_raw.indices.tolist(),
        values=sparse_raw.values.tolist(),
    )
    return dense_vec, sparse_vec


def _hit_to_dict(hit, score_key: str = "score") -> dict:
    payload = hit.payload or {}
    return {
        "chunk_id": payload.get("chunk_id", str(hit.id)),
        "text": payload.get("text", ""),
        "metadata": {
            "job_id": payload.get("job_id", ""),
            "job_title": payload.get("job_title", ""),
            "company_name": payload.get("company_name", ""),
            "job_category": payload.get("job_category", ""),
            "job_level": payload.get("job_level", ""),
            "job_location": payload.get("job_location", ""),
            "publication_date": payload.get("publication_date", ""),
            "tags": payload.get("tags", ""),
        },
        score_key: round(float(hit.score or 0.0), 4),
    }


class Retriever:
    """Retrieve top-N job chunks for a natural-language query."""

    def __init__(self, client: QdrantClient | None = None):
        self._client = client or _get_client()

    def retrieve(
        self,
        query: str,
        top_n: int = DEFAULT_TOP_N,
        mode: SearchMode = "hybrid",
        use_reranker: bool | None = None,
    ) -> list[dict]:
        """
        Parameters
        ----------
        query : str
            Natural-language search query.
        top_n : int
            Number of chunks to return after (optional) reranking.
        mode : {"dense", "sparse", "hybrid"}
            Which vector space(s) to search.
        use_reranker : bool | None
            If True, pull a larger candidate pool then cross-encoder rerank.
            Defaults to config USE_RERANKER.

        Returns
        -------
        list[dict]
            Each item has chunk_id, text, metadata, and score
            (plus rerank_score when reranking is on).
        """
        apply_rerank = USE_RERANKER if use_reranker is None else use_reranker
        # Larger first-stage pool when reranking so the cross-encoder has room.
        pool = max(top_n, RERANK_CANDIDATE_POOL) if apply_rerank else top_n

        dense_vec, sparse_vec = _embed_query(query)

        if mode == "dense":
            hits = self._client.query_points(
                collection_name=COLLECTION_NAME,
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                limit=pool,
                with_payload=True,
            ).points
            candidates = [_hit_to_dict(h, "score") for h in hits]
        elif mode == "sparse":
            hits = self._client.query_points(
                collection_name=COLLECTION_NAME,
                query=sparse_vec,
                using=SPARSE_VECTOR_NAME,
                limit=pool,
                with_payload=True,
            ).points
            candidates = [_hit_to_dict(h, "score") for h in hits]
        else:
            # hybrid — RRF over dense + sparse prefetches
            hits = self._client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    Prefetch(
                        query=dense_vec,
                        using=DENSE_VECTOR_NAME,
                        limit=max(pool, PREFETCH_LIMIT),
                    ),
                    Prefetch(
                        query=sparse_vec,
                        using=SPARSE_VECTOR_NAME,
                        limit=max(pool, PREFETCH_LIMIT),
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=pool,
                with_payload=True,
            ).points
            candidates = [_hit_to_dict(h, "score") for h in hits]

        if apply_rerank and candidates:
            return rerank(query, candidates, top_n=top_n)

        return candidates[:top_n]