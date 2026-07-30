"""Smoke tests for score fusion helpers."""

from rag_core.retrieval.retriever import Retriever


def test_fuse_results_prefers_high_combined_score():
    vector = [
        {
            "chunk_id": "a",
            "text": "a",
            "metadata": {},
            "similarity_score": 0.9,
        },
        {
            "chunk_id": "b",
            "text": "b",
            "metadata": {},
            "similarity_score": 0.5,
        },
    ]
    keyword = [
        {
            "chunk_id": "b",
            "text": "b",
            "metadata": {},
            "bm25_score": 10.0,
        },
        {
            "chunk_id": "c",
            "text": "c",
            "metadata": {},
            "bm25_score": 2.0,
        },
    ]
    fused = Retriever._fuse_results(vector, keyword, top_n=3)
    assert len(fused) == 3
    assert all("score" in r for r in fused)
    # Highest combined should be first
    assert fused[0]["score"] >= fused[1]["score"]
