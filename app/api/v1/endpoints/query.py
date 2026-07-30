from fastapi import APIRouter

from app.services.query_service import QueryService

router = APIRouter()
service = QueryService()


@router.get("/query")
async def query(query: str,
    top_n: int = 30,
    mode: str = "hybrid",
    use_reranker: bool= True,):
    return await service.stream_query(query=query,
        top_n=top_n,
        mode=mode,
        use_reranker=use_reranker,)