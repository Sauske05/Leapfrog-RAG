"""Aggregate v1 endpoint routers."""

from fastapi import APIRouter

from app.api.v1.endpoints import health, query

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(query.router)