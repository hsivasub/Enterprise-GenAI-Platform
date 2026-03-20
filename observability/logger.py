"""
Structured Logging — Observability Layer
=========================================
Provides JSON-structured logging for production with correlation IDs,
request metadata, token usage, latency, and cost tracking baked in.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from config.settings import settings

# Context variable — propagates correlation ID across async call chains
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    return _correlation_id.get() or str(uuid.uuid4())


def set_correlation_id(cid: str) -> None:
    _correlation_id.set(cid)


# ─────────────────────────────────────────────────────────────
# JSON Formatter
# ─────────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.
    This is what log aggregators (Datadog, CloudWatch, Loki) eat natively.
    """

    RESERVED_ATTRS = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_dict: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT.value,
        }

        # Attach structured extra fields (anything not in RESERVED_ATTRS)
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                log_dict[key] = value

        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_dict, default=str)


class PlainFormatter(logging.Formatter):
    """Human-readable formatter used in development."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        cid = get_correlation_id()[:8] if get_correlation_id() else "--------"
        return (
            f"{color}[{ts}][{record.levelname:<8}][{cid}] "
            f"{record.name}: {record.getMessage()}{self.RESET}"
        )


# ─────────────────────────────────────────────────────────────
# Logger factory
# ─────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger. Call once per module.
    In production (STRUCTURED_LOG_FORMAT=True) → JSON output.
    In development → colored plain text.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured — avoid duplicate handlers
        return logger

    logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if settings.STRUCTURED_LOG_FORMAT:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(PlainFormatter())
    logger.addHandler(console_handler)

    # File handler (always JSON for log aggregation)
    try:
        import os
        os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)
        file_handler = logging.FileHandler(settings.LOG_FILE_PATH)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    except Exception:
        pass  # Don't fail startup if log file path is unavailable

    logger.propagate = False
    return logger


# ─────────────────────────────────────────────────────────────
# Specialized log helpers
# ─────────────────────────────────────────────────────────────

class LLMCallLogger:
    """
    Logs every LLM call with prompt, response, tokens, cost, and latency.
    This data is the foundation for cost dashboards and debugging.
    """

    def __init__(self) -> None:
        self._logger = get_logger("llm.calls")

    def log_request(
        self,
        *,
        model: str,
        prompt: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        cost = self._estimate_cost(input_tokens, output_tokens)
        self._logger.info(
            "LLM call completed",
            extra={
                "event_type": "llm_call",
                "model": model,
                "prompt_preview": prompt[:200],          # First 200 chars only
                "response_preview": response[:200],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_ms": round(latency_ms, 2),
                "estimated_cost_usd": round(cost, 6),
                **(metadata or {}),
            },
        )

    @staticmethod
    def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * settings.COST_PER_1K_INPUT_TOKENS / 1000
            + output_tokens * settings.COST_PER_1K_OUTPUT_TOKENS / 1000
        )


class RetrievalLogger:
    """Logs retrieval operations including latency and top-k scores."""

    def __init__(self) -> None:
        self._logger = get_logger("retrieval.calls")

    def log_retrieval(
        self,
        *,
        query: str,
        num_results: int,
        top_score: float,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._logger.info(
            "Retrieval completed",
            extra={
                "event_type": "retrieval",
                "query_preview": query[:200],
                "num_results": num_results,
                "top_similarity_score": round(top_score, 4),
                "latency_ms": round(latency_ms, 2),
                **(metadata or {}),
            },
        )


class RequestLogger:
    """HTTP request/response logger for the API gateway."""

    def __init__(self) -> None:
        self._logger = get_logger("api.requests")

    def log_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
        user_id: Optional[str] = None,
    ) -> None:
        self._logger.info(
            f"{method} {path} → {status_code}",
            extra={
                "event_type": "http_request",
                "method": method,
                "path": path,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 2),
                "user_id": user_id,
            },
        )


# Singletons — import these in other modules
llm_logger = LLMCallLogger()
retrieval_logger = RetrievalLogger()
request_logger = RequestLogger()
