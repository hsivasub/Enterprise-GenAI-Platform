"""
FastAPI Application — API Gateway Main Entry Point
====================================================
The central API gateway that exposes all platform services.

Features:
- OpenAPI docs at /docs and /redoc
- Structured CORS and middleware
- Prometheus metrics at /metrics
- Health check at /health
- Request logging middleware
- Rate limiting (via slowapi)
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from api.routes import chat, ingest
from config.settings import settings
from observability.logger import get_logger, request_logger, set_correlation_id
from observability.metrics import metrics

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Lifespan — startup / shutdown
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Startup: pre-warm components to avoid cold-start latency on first request.
    Shutdown: flush logs, close connections.
    """
    logger.info(
        f"Starting {settings.APP_NAME} v{settings.APP_VERSION}",
        extra={"environment": settings.ENVIRONMENT.value},
    )

    # Pre-warm the embedding model (downloads ~80MB on first use)
    try:
        from api.dependencies import _get_embedder, _get_vector_store
        _get_vector_store()   # This also initializes the embedder
        logger.info("Vector store initialized")
    except Exception as exc:
        logger.warning(f"Vector store pre-warm failed (non-fatal): {exc}")

    yield  # Application is running

    logger.info("Shutting down — flushing resources")


# ─────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Enterprise GenAI Platform — AI Financial Assistant\n\n"
            "Provides RAG-powered Q&A, multi-step agent reasoning, "
            "structured data queries, and responsible AI guardrails."
        ),
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        docs_url=f"{settings.API_PREFIX}/docs",
        redoc_url=f"{settings.API_PREFIX}/redoc",
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging + correlation IDs ───────────────────────
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        set_correlation_id(request_id)
        start = time.perf_counter()

        response: Response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        request_logger.log_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=elapsed_ms,
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(round(elapsed_ms, 1))
        return response

    # ── Routes ──────────────────────────────────────────────────
    app.include_router(chat.router, prefix=settings.API_PREFIX)
    app.include_router(ingest.router, prefix=settings.API_PREFIX)

    # ── Utility endpoints ────────────────────────────────────────

    @app.get("/health", tags=["System"], summary="Platform health check")
    async def health():
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT.value,
        }

    @app.get("/metrics", tags=["System"], response_class=PlainTextResponse,
             summary="Prometheus metrics")
    async def prometheus_metrics():
        return metrics.get_prometheus_text()

    @app.get("/metrics/json", tags=["System"], summary="JSON metrics snapshot")
    async def json_metrics():
        snap = metrics.get_snapshot()
        return {
            "total_requests": snap.total_requests,
            "total_llm_calls": snap.total_llm_calls,
            "total_cost_usd": round(snap.total_cost_usd, 4),
            "cache_hits": snap.cache_hits,
            "cache_misses": snap.cache_misses,
            "avg_latency_ms": round(snap.avg_latency_ms, 1),
            "errors": snap.errors,
        }

    @app.get(f"{settings.API_PREFIX}/prompts", tags=["System"],
             summary="List all prompt versions")
    async def list_prompts():
        from prompts.templates import prompt_registry
        return prompt_registry.list_prompts()

    # ── Global exception handler ─────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            f"Unhandled exception: {exc}",
            exc_info=True,
            extra={"path": request.url.path},
        )
        metrics.record_request(latency_ms=0, success=False)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.DEBUG else "Contact support",
            },
        )

    return app


# ─────────────────────────────────────────────────────────────
# App instance (imported by uvicorn / gunicorn)
# ─────────────────────────────────────────────────────────────

app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.ENVIRONMENT.value == "development",
        log_config=None,   # Use our custom logger
        workers=1,
    )
