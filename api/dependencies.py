"""
Dependency Injection — API Layer
===================================
FastAPI dependency functions that wire together the platform components.
Using DI ensures testability: swap real components with mocks in tests.
"""

from __future__ import annotations

import functools
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from agents.agent import AgentOrchestrator
from agents.llm_client import get_llm_client
from config.settings import settings
from guardrails.input_validator import InputValidator
from guardrails.output_filter import OutputFilter
from ingestion.embedder import ChunkEmbedder
from ingestion.vector_store import FAISSVectorStore
from observability.logger import get_logger
from retrieval.retriever import RetrieverEngine

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Component singletons (lazy-initialized)
# ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_vector_store() -> FAISSVectorStore:
    embedder = _get_embedder()
    return FAISSVectorStore(dimension=embedder.dimension)


@functools.lru_cache(maxsize=1)
def _get_embedder() -> ChunkEmbedder:
    return ChunkEmbedder()


@functools.lru_cache(maxsize=1)
def _get_retriever_engine() -> RetrieverEngine:
    vs = _get_vector_store()
    emb = _get_embedder()
    return RetrieverEngine(vector_store=vs, embedder=emb)


@functools.lru_cache(maxsize=1)
def _get_redis_client():
    """Return Redis client or None if unavailable."""
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL)
        client.ping()
        logger.info("Redis connected")
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}. Caching disabled.")
        return None


@functools.lru_cache(maxsize=1)
def _get_agent_orchestrator() -> AgentOrchestrator:
    retriever = _get_retriever_engine()
    llm = get_llm_client()
    redis_client = _get_redis_client()
    return AgentOrchestrator(
        retriever_engine=retriever,
        llm_client=llm,
        redis_client=redis_client,
    )


# ─────────────────────────────────────────────────────────────
# FastAPI dependency functions
# ─────────────────────────────────────────────────────────────

def get_agent() -> AgentOrchestrator:
    return _get_agent_orchestrator()


def get_retriever() -> RetrieverEngine:
    return _get_retriever_engine()


def get_vector_store() -> FAISSVectorStore:
    return _get_vector_store()


def get_input_validator() -> InputValidator:
    return InputValidator()


def get_output_filter() -> OutputFilter:
    return OutputFilter()


def verify_api_key(request: Request) -> str:
    """API key authentication middleware dependency."""
    api_key = request.headers.get(settings.API_KEY_HEADER)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header.",
        )
    if api_key != settings.PLATFORM_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return api_key
