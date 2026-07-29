"""
LlamaIndex BaseRetriever adapter around our Qdrant hybrid Retriever.

Applies optional cross-encoder reranking after first-stage retrieval.
"""

from __future__ import annotations

from typing import List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag_core.config import DEFAULT_TOP_N, USE_RERANKER
from rag_core.retrieval.retriever import Retriever, SearchMode


class QdrantHybridRetriever(BaseRetriever):
    """Expose rag_core.retrieval.Retriever as a LlamaIndex retriever."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        top_n: int = DEFAULT_TOP_N,
        mode: SearchMode = "hybrid",
        use_reranker: bool | None = None,
    ):
        super().__init__()
        self._retriever = retriever or Retriever()
        self.top_n = top_n
        self.mode = mode
        self.use_reranker = USE_RERANKER if use_reranker is None else use_reranker

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        hits = self._retriever.retrieve(
            query,
            top_n=self.top_n,
            mode=self.mode,
            use_reranker=self.use_reranker,
        )

        nodes: List[NodeWithScore] = []
        for hit in hits:
            meta = hit.get("metadata") or {}
            score = hit.get("rerank_score", hit.get("score"))
            node = TextNode(
                text=hit.get("text", ""),
                id_=hit.get("chunk_id", ""),
                metadata={
                    "chunk_id": hit.get("chunk_id", ""),
                    "job_id": meta.get("job_id", ""),
                    "job_title": meta.get("job_title", ""),
                    "company_name": meta.get("company_name", ""),
                    "job_category": meta.get("job_category", ""),
                    "job_level": meta.get("job_level", ""),
                    "job_location": meta.get("job_location", ""),
                    "publication_date": meta.get("publication_date", ""),
                    "tags": meta.get("tags", ""),
                    "score": hit.get("score"),
                    "rerank_score": hit.get("rerank_score"),
                },
            )
            nodes.append(NodeWithScore(node=node, score=float(score or 0.0)))
        return nodes