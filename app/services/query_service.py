"""
High-level query service: invokes rag_core generation and streams SSE events.

"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Literal, Optional

from fastapi import HTTPException

from rag_core.generation.query_engine import answer_query


class QueryService:
    async def stream_query(
        self,
        query: str,
        top_n: int = 5,
        mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
        use_reranker: Optional[bool] = None,
    ):

        # yield {
        #     "event": "status",
        #     "data": {"stage": "started", "query": query},
        # }

        try:
        #     yield {
        #         "event": "status",
        #         "data": {"stage": "retrieving_and_generating"},
        #     }

            result = await asyncio.to_thread(
                answer_query,
                query,
                top_n,
                mode,
                use_reranker,
            )

            # yield {
            #     "event": "answer",
            #     "data": {
            #         "query": result.get("query", query),
            #         "answer": result.get("answer", ""),
            #         "source_nodes": result.get("source_nodes", []),
            #         "use_reranker": result.get("use_reranker", False),
            #     },
            # }

            # yield {
            #     "event": "done",
            #     "data": {"ok": True},
            # }
            print(result)
            print(f'This is the answer: {result.get("answer", "")}')
            return result.get("answer", "")

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc

    def query_sync(
        self,
        query: str,
        top_n: int = 5,
        mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
        use_reranker: Optional[bool] = None,
    ) -> dict:
        return answer_query(
            query=query,
            top_n=top_n,
            mode=mode,
            use_reranker=use_reranker,
        )