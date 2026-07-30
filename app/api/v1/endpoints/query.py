from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.services.query_service import QueryService

router = APIRouter()
service = QueryService()


@router.get("/stream")
async def stream(query: str):
    return EventSourceResponse(
        service.stream_query(query),
        ping=15,  # keep connection alive
    )