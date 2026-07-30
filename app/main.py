"""
FastAPI entrypoint.

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import API_V1_PREFIX
from rag_core.runtime import init_runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load embedding models, Qdrant client, reranker, Gemini once.
    init_runtime(load_reranker=True, load_llm=True)
    yield


app = FastAPI(
    title="LF Jobs RAG API",
    description="Hybrid RAG API for job listing search (Qdrant + Gemini).",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix=API_V1_PREFIX)


@app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "message": "LF Jobs RAG API",
        "docs": "/docs",
        "api": API_V1_PREFIX,
    }