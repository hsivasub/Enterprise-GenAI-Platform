"""
Retrieval Engine — Retrieval Layer
=====================================
Implements:
1. Semantic (dense) retrieval via FAISS
2. Optional hybrid retrieval (dense + BM25 sparse)
3. Optional cross-encoder reranking

This is intentionally decoupled from the agent — the agent calls
the retriever as a tool, keeping concerns separated.
"""

from __future__ import annotations

import time
from typing import List, Optional

from ingestion.chunker import DocumentChunk
from ingestion.embedder import ChunkEmbedder
from ingestion.vector_store import FAISSVectorStore, SearchResult
from config.settings import settings
from observability.logger import get_logger, retrieval_logger
from observability.tracer import create_tracer

logger = get_logger(__name__)
tracer = create_tracer("retrieval")


class RetrievalConfig:
    """Runtime-configurable retrieval parameters."""
    def __init__(
        self,
        top_k: int = settings.FAISS_TOP_K,
        min_score: float = 0.3,
        enable_reranking: bool = settings.ENABLE_RERANKING,
        enable_hybrid: bool = settings.ENABLE_HYBRID_SEARCH,
    ) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self.enable_reranking = enable_reranking
        self.enable_hybrid = enable_hybrid


class BM25Retriever:
    """
    Sparse keyword retrieval using BM25.
    Complements dense embeddings for exact-match queries (e.g., ticker symbols).
    Requires: pip install rank_bm25
    """

    def __init__(self, chunks: List[DocumentChunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 not installed: pip install rank-bm25")

        self._chunks = chunks
        tokenized = [chunk.content.lower().split() for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]


class CrossEncoderReranker:
    """
    Reranks top-k results using a cross-encoder for higher precision.
    Slower but significantly more accurate for complex queries.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(self, model_name: str = settings.RERANKER_MODEL) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("sentence-transformers not installed")
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        if not results:
            return results
        pairs = [(query, r.chunk.content) for r in results]
        scores = self._model.predict(pairs)
        reranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [SearchResult(chunk=r.chunk, score=float(s)) for r, s in reranked]


class RetrieverEngine:
    """
    Main retrieval orchestrator used by the agent and API layer.

    Supports:
    - Dense-only: fastest, good for semantic questions
    - Hybrid: dense + BM25, better recall, needed for keyword queries
    - With reranking: highest precision, higher latency
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: ChunkEmbedder,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._config = config or RetrievalConfig()
        self._reranker: Optional[CrossEncoderReranker] = None

        if self._config.enable_reranking:
            try:
                self._reranker = CrossEncoderReranker()
                logger.info("Cross-encoder reranker loaded")
            except Exception as e:
                logger.warning(f"Reranker unavailable: {e}. Falling back to dense-only.")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> List[SearchResult]:
        """
        Main retrieval method. Returns ranked list of SearchResult.

        Args:
            query: Natural language question
            top_k: Override default top_k
            filters: Metadata filters (e.g., {"doc_type": "pdf"})
        """
        k = top_k or self._config.top_k
        start = time.perf_counter()

        with tracer.start_span("retrieve", attributes={"query_len": len(query), "top_k": k}):
            # 1. Embed query
            query_embedding = self._embedder.embed_query(query)

            # 2. Dense retrieval
            results = self._store.search(query_embedding, top_k=k * 2)

            # 3. Metadata filtering (post-filter)
            if filters:
                results = self._apply_filters(results, filters)

            # 4. Hybrid search (merge BM25 + dense)
            if self._config.enable_hybrid:
                results = self._hybrid_search(query, results, k)

            # 5. Reranking
            if self._reranker and results:
                results = self._reranker.rerank(query, results)

            # 6. Limit final results
            results = results[:k]

        elapsed_ms = (time.perf_counter() - start) * 1000
        top_score = results[0].score if results else 0.0

        retrieval_logger.log_retrieval(
            query=query,
            num_results=len(results),
            top_score=top_score,
            latency_ms=elapsed_ms,
        )

        return results

    def get_context_for_prompt(
        self, query: str, max_tokens: int = settings.RAG_MAX_CONTEXT_TOKENS
    ) -> str:
        """
        Retrieve and format context for injection into an LLM prompt.
        Respects max token budget by truncating lower-scored chunks first.
        """
        results = self.retrieve(query)
        context_parts: List[str] = []
        total_tokens = 0
        tokens_per_char = 0.25  # Rough estimate: 4 chars ≈ 1 token

        for i, result in enumerate(results):
            chunk_tokens = int(len(result.chunk.content) * tokens_per_char)
            if total_tokens + chunk_tokens > max_tokens:
                break
            source = result.chunk.metadata.get("filename", "unknown")
            context_parts.append(
                f"[Source {i+1}: {source} | Relevance: {result.score:.2f}]\n"
                f"{result.chunk.content}"
            )
            total_tokens += chunk_tokens

        return "\n\n---\n\n".join(context_parts)

    @staticmethod
    def _apply_filters(
        results: List[SearchResult], filters: dict
    ) -> List[SearchResult]:
        """Filter results by metadata key-value pairs."""
        filtered = []
        for r in results:
            match = all(
                r.chunk.metadata.get(k) == v for k, v in filters.items()
            )
            if match:
                filtered.append(r)
        return filtered

    def _hybrid_search(
        self,
        query: str,
        dense_results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion (RRF) to merge dense + sparse results.
        RRF is more stable than score-level combination.
        """
        all_chunks = list(self._store._id_to_chunk.values())
        if not all_chunks:
            return dense_results

        try:
            bm25 = BM25Retriever(all_chunks)
            sparse_results = bm25.search(query, top_k * 2)
        except Exception:
            return dense_results  # Graceful degradation

        # RRF: score = sum(1 / (rank + 60))
        scores: dict = {}
        for rank, r in enumerate(dense_results):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + 1 / (rank + 60)

        for rank, r in enumerate(sparse_results):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + 1 / (rank + 60)

        chunk_map = {r.chunk.chunk_id: r for r in dense_results + sparse_results}
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

        return [
            SearchResult(chunk=chunk_map[cid].chunk, score=scores[cid])
            for cid in sorted_ids
            if cid in chunk_map
        ]
