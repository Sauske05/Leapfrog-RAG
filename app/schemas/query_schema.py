"""Request / response schemas for the query API."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from rag_core.config import DEFAULT_TOP_N


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language job search query.")
    top_n: int = Field(DEFAULT_TOP_N, ge=1, le=20)
    mode: Literal["dense", "sparse", "hybrid"] = Field(
        "hybrid", description="Retrieval mode."
    )
    use_reranker: Optional[bool] = Field(
        None, description="Override default USE_RERANKER. Null = use config default."
    )


class SourceNode(BaseModel):
    chunk_id: str
    score: float = 0.0
    rerank_score: Optional[float] = None
    retrieval_score: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    text: str = ""


class QueryResponse(BaseModel):
    query: str
    answer: str
    source_nodes: list[SourceNode] = Field(default_factory=list)
    use_reranker: bool = False