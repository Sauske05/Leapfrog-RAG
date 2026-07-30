"""GET /api/v1/health"""

from fastapi import APIRouter

from rag_core.runtime import runtime

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "runtime_ready": runtime.ready,
        "has_llm": runtime.llm is not None,
        "has_dense": runtime.dense_model is not None,
        "has_sparse": runtime.sparse_model is not None,
        "has_reranker": runtime.reranker is not None,
    }