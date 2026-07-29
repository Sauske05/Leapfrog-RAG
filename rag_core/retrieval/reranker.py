"""
Optional cross-encoder reranker (enhancement, off by default).

A cross-encoder scores (query, chunk) pairs jointly, which is more
accurate than comparing separately-computed embeddings, at the cost of
being slower -- so we only run it on the small candidate pool returned
by the first-stage retriever, not the whole corpus.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag_core.config import RERANKER_MODEL_NAME


@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    """Load (and cache) the cross-encoder reranking model."""
    return CrossEncoder(RERANKER_MODEL_NAME)


def rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """Rerank retrieved candidates by cross-encoder relevance to the query.

    Parameters
    ----------
    query : str
        The user's natural-language query.
    candidates : list[dict]
        Candidate chunks (each must have a "text" key), typically the
        output of the hybrid retriever.
    top_n : int
        Number of top-ranked results to keep after reranking.

    Returns
    -------
    list[dict]
        The candidates, reordered by relevance and truncated to top_n,
        each with an added "rerank_score" key.
    """
    if not candidates:
        return []

    model = get_reranker_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = round(float(score), 4)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_n]