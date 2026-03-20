"""
Embedding Generation — Ingestion Layer
========================================
Supports multiple embedding providers behind a unified interface:
- SentenceTransformers (local, no API cost) — default
- OpenAI text-embedding-3-small/large
- Azure OpenAI embeddings

Design decisions:
- Batched processing for throughput (configurable batch size)
- L2-normalized embeddings for cosine similarity FAISS index
- Providers are interchangeable via config — no code changes needed
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from config.settings import EmbeddingProvider, settings
from ingestion.chunker import DocumentChunk
from observability.logger import get_logger
from observability.metrics import metrics

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Provider interface
# ─────────────────────────────────────────────────────────────

class EmbedderBase(ABC):
    """Abstract embedding provider."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return (N, D) float32 array of normalized embeddings."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns 1-D array."""
        return self.embed_texts([text])[0]


# ─────────────────────────────────────────────────────────────
# SentenceTransformers (local)
# ─────────────────────────────────────────────────────────────

class SentenceTransformerEmbedder(EmbedderBase):
    """
    Runs a SentenceTransformer model locally.
    First call downloads the model to ~/.cache/huggingface.
    Default: all-MiniLM-L6-v2 (384-dim, ~80MB, very fast on CPU).
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "SentenceTransformers not installed: pip install sentence-transformers"
            )
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready. Dimension: {self._dim}")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        start = time.perf_counter()
        embeddings = self._model.encode(
            texts,
            batch_size=settings.INGESTION_BATCH_SIZE,
            normalize_embeddings=True,   # L2 normalization for cosine similarity
            show_progress_bar=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            f"Embedded {len(texts)} texts",
            extra={"model": self._model_name, "latency_ms": round(elapsed_ms, 1)},
        )
        return np.array(embeddings, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# OpenAI Embeddings
# ─────────────────────────────────────────────────────────────

class OpenAIEmbedder(EmbedderBase):
    """
    Uses OpenAI text-embedding-3-small/large API.
    Costs ~$0.02 per 1M tokens — use batching to minimize API calls.
    """

    # Dimension for text-embedding-3-small
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = settings.OPENAI_EMBEDDING_MODEL) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("openai not installed: pip install openai")

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")

        self._client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = model
        self._dim = self.DIMENSIONS.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # OpenAI recommends batches of up to 2048 inputs per call
        batch_size = 100
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            start = time.perf_counter()
            response = self._client.embeddings.create(model=self._model, input=batch)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Record token usage
            total_tokens = response.usage.total_tokens
            metrics.record_llm_call(self._model, total_tokens, 0)

            logger.debug(
                f"OpenAI embedding batch: {len(batch)} texts",
                extra={"model": self._model, "tokens": total_tokens, "latency_ms": elapsed_ms},
            )
            for item in sorted(response.data, key=lambda x: x.index):
                all_embeddings.append(item.embedding)

        embeddings = np.array(all_embeddings, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.where(norms == 0, 1, norms)


# ─────────────────────────────────────────────────────────────
# Azure OpenAI Embeddings
# ─────────────────────────────────────────────────────────────

class AzureOpenAIEmbedder(EmbedderBase):
    """Azure OpenAI embedding endpoint — for enterprise Azure customers."""

    def __init__(self) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("openai not installed: pip install openai")

        if not settings.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY required")

        self._client = openai.AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        self._deployment = settings.AZURE_OPENAI_DEPLOYMENT or "text-embedding-3-small"
        self._dim = 1536

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        response = self._client.embeddings.create(
            model=self._deployment, input=texts
        )
        embeddings = np.array(
            [item.embedding for item in sorted(response.data, key=lambda x: x.index)],
            dtype=np.float32,
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.where(norms == 0, 1, norms)


# ─────────────────────────────────────────────────────────────
# Chunk Embedder (wraps chunks → embeddings)
# ─────────────────────────────────────────────────────────────

class ChunkEmbedder:
    """
    Adds embeddings to DocumentChunks.
    Handles batch processing and error recovery.
    """

    def __init__(self, provider: Optional[EmbedderBase] = None) -> None:
        self._provider = provider or _build_default_provider()

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[tuple[DocumentChunk, np.ndarray]]:
        """
        Returns list of (chunk, embedding_vector) pairs.
        Batches embedding calls for efficiency.
        """
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        start = time.perf_counter()
        embeddings = self._provider.embed_texts(texts)
        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            f"Embedded {len(chunks)} chunks",
            extra={"latency_ms": round(elapsed, 1), "dim": embeddings.shape[1]},
        )
        return list(zip(chunks, embeddings))

    def embed_query(self, query: str) -> np.ndarray:
        return self._provider.embed_query(query)


def _build_default_provider() -> EmbedderBase:
    """Build embedding provider based on settings."""
    provider = settings.EMBEDDING_PROVIDER
    if provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbedder()
    elif provider == EmbeddingProvider.AZURE_OPENAI:
        return AzureOpenAIEmbedder()
    else:
        return SentenceTransformerEmbedder()


# Optional import for type hints
from typing import Optional
