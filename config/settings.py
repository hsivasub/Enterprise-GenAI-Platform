"""
Enterprise GenAI Platform - Central Configuration
==================================================
All settings are driven by environment variables with sensible defaults.
Pydantic BaseSettings handles parsing, type checking, and .env loading.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"  # Ollama or similar


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    AZURE_OPENAI = "azure_openai"


class Settings(BaseSettings):
    """Central settings class — loaded once and cached."""

    # ──────────────────────────────────────────────
    # Application
    # ──────────────────────────────────────────────
    APP_NAME: str = "Enterprise GenAI Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ──────────────────────────────────────────────
    # API Server
    # ──────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["*"]
    API_KEY_HEADER: str = "X-API-Key"
    PLATFORM_API_KEY: str = "dev-secret-key-change-in-production"

    # ──────────────────────────────────────────────
    # LLM Provider
    # ──────────────────────────────────────────────
    LLM_PROVIDER: LLMProvider = LLMProvider.OPENAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_TOKENS: int = 2048
    OPENAI_TEMPERATURE: float = 0.0

    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"

    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"

    # Local/Ollama
    LOCAL_LLM_BASE_URL: str = "http://localhost:11434"
    LOCAL_LLM_MODEL: str = "llama3"

    # ──────────────────────────────────────────────
    # Embedding
    # ──────────────────────────────────────────────
    EMBEDDING_PROVIDER: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ──────────────────────────────────────────────
    # Vector Store (FAISS)
    # ──────────────────────────────────────────────
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    FAISS_TOP_K: int = 5
    ENABLE_HYBRID_SEARCH: bool = False

    # Pinecone (optional)
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "genai-platform"

    # ──────────────────────────────────────────────
    # Database (PostgreSQL)
    # ──────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://genai:genai@localhost:5432/genai_db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_ECHO: bool = False

    # DuckDB (analytical queries)
    DUCKDB_PATH: str = "./data/analytical.duckdb"

    # ──────────────────────────────────────────────
    # Redis (caching)
    # ──────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL_SECONDS: int = 3600          # 1 hour default cache TTL
    REDIS_MAX_CONNECTIONS: int = 20
    ENABLE_RESPONSE_CACHE: bool = True

    # ──────────────────────────────────────────────
    # MLflow (experiment tracking)
    # ──────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "genai-platform"
    MLFLOW_ARTIFACT_ROOT: str = "./mlruns"

    # ──────────────────────────────────────────────
    # Document Ingestion
    # ──────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MAX_DOCUMENT_SIZE_MB: int = 50
    SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".txt", ".md", ".docx"]
    INGESTION_BATCH_SIZE: int = 32         # embeddings batched for efficiency

    # ──────────────────────────────────────────────
    # RAG Pipeline
    # ──────────────────────────────────────────────
    RAG_MAX_CONTEXT_TOKENS: int = 6000
    RAG_SYSTEM_PROMPT_VERSION: str = "v1"
    ENABLE_RERANKING: bool = False
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ──────────────────────────────────────────────
    # Agent Framework
    # ──────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_TIMEOUT_SECONDS: int = 120
    ENABLE_AGENT_MEMORY: bool = True
    AGENT_VERBOSE: bool = False

    # ──────────────────────────────────────────────
    # Guardrails
    # ──────────────────────────────────────────────
    ENABLE_PII_DETECTION: bool = True
    ENABLE_CONTENT_FILTER: bool = True
    ENABLE_HALLUCINATION_CHECK: bool = False  # Placeholder — expensive
    MAX_INPUT_LENGTH: int = 4096
    BLOCKED_TERMS: List[str] = ["<script>", "DROP TABLE", "rm -rf"]

    # ──────────────────────────────────────────────
    # Cost Tracking
    # ──────────────────────────────────────────────
    # Prices in USD per 1K tokens  (as of 2024 — update periodically)
    COST_PER_1K_INPUT_TOKENS: float = 0.00015   # gpt-4o-mini input
    COST_PER_1K_OUTPUT_TOKENS: float = 0.0006   # gpt-4o-mini output
    MONTHLY_BUDGET_USD: float = 100.0

    # ──────────────────────────────────────────────
    # Observability
    # ──────────────────────────────────────────────
    ENABLE_TRACING: bool = True
    JAEGER_ENDPOINT: Optional[str] = None       # e.g. http://jaeger:14268/api/traces
    PROMETHEUS_PORT: int = 9090
    STRUCTURED_LOG_FORMAT: bool = True          # JSON logs in production
    LOG_FILE_PATH: str = "./logs/platform.log"

    # ──────────────────────────────────────────────
    # Service URLs (microservice mode)
    # ──────────────────────────────────────────────
    RETRIEVAL_SERVICE_URL: str = "http://localhost:8001"
    AGENT_SERVICE_URL: str = "http://localhost:8002"
    EVALUATION_SERVICE_URL: str = "http://localhost:8003"

    # ──────────────────────────────────────────────
    # Rate Limiting
    # ──────────────────────────────────────────────
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_REQUESTS_PER_DAY: int = 10000

    @field_validator("OPENAI_API_KEY", mode="before")
    @classmethod
    def validate_openai_key(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.startswith("sk-"):
            # Accept project keys (sk-proj-) and service account keys
            if not (v.startswith("sk-proj-") or v.startswith("sk-svcacct-")):
                pass  # Allow — key format may vary
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return cached Settings singleton.
    Using lru_cache ensures we parse env vars only once at startup.
    Call `get_settings.cache_clear()` in tests to reset.
    """
    return Settings()


# Convenience alias used throughout the codebase
settings = get_settings()
