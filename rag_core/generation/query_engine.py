"""
LlamaIndex query engine: Qdrant hybrid retrieval (+ rerank) + Gemini generation.
"""

from __future__ import annotations

from typing import Any, Optional

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)

from rag_core.config import DEFAULT_TOP_N, USE_RERANKER
from rag_core.generation.llm import get_llm
from rag_core.generation.prompts import QA_PROMPT_TEMPLATE, SYSTEM_PROMPT
from rag_core.retrieval.llama_index_retriever import QdrantHybridRetriever
from rag_core.retrieval.retriever import Retriever, SearchMode


def build_query_engine(
    top_n: int = DEFAULT_TOP_N,
    mode: SearchMode = "hybrid",
    use_reranker: bool | None = None,
    retriever: Optional[Retriever] = None,
) -> RetrieverQueryEngine:
    """Wire hybrid retriever (+ optional rerank) + Gemini into a query engine."""
    apply_rerank = USE_RERANKER if use_reranker is None else use_reranker

    li_retriever = QdrantHybridRetriever(
        retriever=retriever,
        top_n=top_n,
        mode=mode,
        use_reranker=apply_rerank,
    )
    llm = get_llm()

    qa_prompt = PromptTemplate(QA_PROMPT_TEMPLATE)
    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=ResponseMode.COMPACT,
        text_qa_template=qa_prompt,
    )

    return RetrieverQueryEngine(
        retriever=li_retriever,
        response_synthesizer=synthesizer,
    )


def answer_query(
    query: str,
    top_n: int = DEFAULT_TOP_N,
    mode: SearchMode = "hybrid",
    use_reranker: bool | None = True,
) -> dict[str, Any]:
    """
    End-to-end: retrieve → (rerank) → generate.

    Returns
    -------
    dict with keys: query, answer, source_nodes, use_reranker
    """
    apply_rerank = USE_RERANKER if use_reranker is None else use_reranker
    engine = build_query_engine(
        top_n=top_n, mode=mode, use_reranker=apply_rerank
    )
    response = engine.query(query)

    sources = []
    for sn in getattr(response, "source_nodes", []) or []:
        meta = dict(sn.node.metadata or {})
        sources.append(
            {
                "chunk_id": meta.get("chunk_id", sn.node.node_id),
                "score": float(sn.score or meta.get("rerank_score") or meta.get("score") or 0.0),
                "rerank_score": meta.get("rerank_score"),
                "retrieval_score": meta.get("score"),
                "metadata": meta,
                "text": sn.node.get_content()[:500],
            }
        )

    return {
        "query": query,
        "answer": str(response),
        "source_nodes": sources,
        "use_reranker": apply_rerank,
        "system_prompt_note": SYSTEM_PROMPT[:80] + "...",
    }
