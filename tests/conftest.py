"""
Shared pytest fixtures and configuration.
=========================================
conftest.py is automatically loaded by pytest — all fixtures defined
here are available in every test module without explicit imports.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

# ── Force test environment before any platform imports ──────────
# This ensures settings.py doesn't try to connect to real services
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-for-tests-only")
os.environ.setdefault("PLATFORM_API_KEY", "test-api-key")
os.environ.setdefault("STRUCTURED_LOG_FORMAT", "false")
os.environ.setdefault("ENABLE_RESPONSE_CACHE", "false")
os.environ.setdefault("ENABLE_PII_DETECTION", "true")
os.environ.setdefault("ENABLE_CONTENT_FILTER", "true")
os.environ.setdefault("MLFLOW_TRACKING_URI", "memory://")  # In-memory MLflow
os.environ.setdefault("LOG_FILE_PATH", "/tmp/test_platform.log")


# ── Sample text fixtures ─────────────────────────────────────────

SAMPLE_FINANCIAL_TEXT = """
Apple Inc. reported total net revenue of $85.8 billion for Q3 2024,
representing a 5% increase year-over-year. Net income was $21.4 billion
with earnings per share of $1.40. Services revenue reached $24.2 billion,
growing 14% year-over-year, driven by App Store, iCloud, and Apple TV+.
The gross margin improved to 46.3%, the highest in company history for Q3.
iPhone revenue was $39.3 billion, while Mac revenue came in at $7.0 billion.
The company returned $26 billion to shareholders through buybacks.
"""

SAMPLE_LONG_TEXT = SAMPLE_FINANCIAL_TEXT * 10  # For chunking tests

SAMPLE_PII_TEXT = (
    "Contact John at john.doe@example.com or call 555-123-4567. "
    "His SSN is 123-45-6789 and card number is 4111-1111-1111-1111."
)


# ── Temp directory fixture ───────────────────────────────────────

@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Temporary directory that is cleaned up after each test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ── Sample document file fixtures ────────────────────────────────

@pytest.fixture
def sample_txt_file(tmp_dir: Path) -> Path:
    """Write sample text to a temp .txt file and return the path."""
    f = tmp_dir / "earnings.txt"
    f.write_text(SAMPLE_FINANCIAL_TEXT, encoding="utf-8")
    return f


@pytest.fixture
def sample_md_file(tmp_dir: Path) -> Path:
    """Sample markdown file."""
    content = "# Earnings Report\n\n" + SAMPLE_FINANCIAL_TEXT
    f = tmp_dir / "report.md"
    f.write_text(content, encoding="utf-8")
    return f


# ── Embedding helpers ────────────────────────────────────────────

EMBEDDING_DIM = 8  # Tiny dimension for fast tests (no real model needed)


def make_random_embedding(dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Generate a normalized random embedding vector."""
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def random_embedding() -> np.ndarray:
    return make_random_embedding()


# ── FastAPI test client ───────────────────────────────────────────

@pytest.fixture
def api_client():
    """
    Returns a FastAPI TestClient with real routes but mocked heavy deps.
    Avoids loading the embedding model and connecting to Redis/PostgreSQL.
    """
    from unittest.mock import MagicMock, patch

    # Patch out expensive singletons
    mock_agent = MagicMock()
    mock_agent.run.return_value = MagicMock(
        answer="Apple revenue was $85.8 billion in Q3 2024.",
        question="What was Apple revenue?",
        tool_calls=[],
        observations=[],
        iterations=1,
        latency_ms=120.0,
        cached=False,
        error=None,
        session_id="test-session",
    )

    mock_vector_store = MagicMock()
    mock_vector_store.total_vectors = 42
    mock_vector_store.index_path = "/tmp/test_faiss"

    with patch("api.dependencies._get_agent_orchestrator", return_value=mock_agent), \
         patch("api.dependencies._get_vector_store", return_value=mock_vector_store):
        from fastapi.testclient import TestClient
        from api.main import app
        yield TestClient(app)
