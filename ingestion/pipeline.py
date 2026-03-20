"""
Ingestion Pipeline Orchestrator
=================================
Ties together: load → chunk → embed → store

This is the entry point for adding new documents to the knowledge base.
Designed to be called from:
1. The API ingest endpoint (one document at a time)
2. Airflow batch ingestion DAGs (directory-level)
3. CLI scripts

MLflow experiment tracking is integrated to record every ingestion run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config.settings import settings
from ingestion.chunker import DocumentChunk, get_chunker
from ingestion.document_loader import DocumentLoader, RawDocument
from ingestion.embedder import ChunkEmbedder
from ingestion.vector_store import FAISSVectorStore
from observability.logger import get_logger
from observability.tracer import create_tracer

logger = get_logger(__name__)
tracer = create_tracer("ingestion")


@dataclass
class IngestionResult:
    """Result returned from an ingestion run."""
    doc_id: str
    filename: str
    chunks_created: int
    chunks_added: int           # May be < chunks_created due to deduplication
    total_chars: int
    elapsed_seconds: float
    strategy: str
    success: bool
    error: Optional[str] = None


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Configuration is fully driven by settings.py, enabling:
    - Zero-code strategy switching (set CHUNKING_STRATEGY env var)
    - A/B testing of chunk sizes via MLflow parameter logging
    """

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        embedder: Optional[ChunkEmbedder] = None,
        chunking_strategy: str = "recursive_character",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self._loader = DocumentLoader()
        self._chunker = get_chunker(
            strategy=chunking_strategy,
            chunk_size=chunk_size or settings.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        )
        self._embedder = embedder or ChunkEmbedder()
        self._vector_store = vector_store or FAISSVectorStore(
            dimension=self._embedder.dimension
        )
        self._strategy = chunking_strategy

    def ingest_file(self, file_path: str | Path) -> IngestionResult:
        """
        Ingest a single file through the full pipeline.
        Idempotent: re-ingesting the same file only adds new chunks.
        """
        path = Path(file_path)
        start = time.perf_counter()

        try:
            with tracer.start_span("ingest_file", attributes={"file": path.name}) as span:
                # 1. Load
                logger.info(f"Ingesting file: {path.name}")
                raw_doc = self._loader.load(path)
                span.set_attribute("doc_id", raw_doc.doc_id)

                # 2. Chunk
                with tracer.start_span("chunk_document", parent_span=span):
                    chunks = self._chunker.chunk(raw_doc)
                    logger.info(
                        f"Chunked into {len(chunks)} chunks",
                        extra={"strategy": self._strategy},
                    )

                # 3. Embed
                with tracer.start_span("embed_chunks", parent_span=span):
                    chunk_emb_pairs = self._embedder.embed_chunks(chunks)

                # 4. Store
                with tracer.start_span("store_vectors", parent_span=span):
                    added = self._vector_store.add_chunks(chunk_emb_pairs)
                    self._vector_store.save()

                elapsed = time.perf_counter() - start
                result = IngestionResult(
                    doc_id=raw_doc.doc_id,
                    filename=path.name,
                    chunks_created=len(chunks),
                    chunks_added=added,
                    total_chars=len(raw_doc.content),
                    elapsed_seconds=round(elapsed, 2),
                    strategy=self._strategy,
                    success=True,
                )
                self._log_to_mlflow(result)
                logger.info(
                    f"Ingestion complete: {path.name}",
                    extra={
                        "chunks_added": added,
                        "elapsed_s": result.elapsed_seconds,
                    },
                )
                return result

        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                f"Ingestion failed: {path.name} — {exc}",
                exc_info=True,
            )
            return IngestionResult(
                doc_id="",
                filename=path.name,
                chunks_created=0,
                chunks_added=0,
                total_chars=0,
                elapsed_seconds=round(elapsed, 2),
                strategy=self._strategy,
                success=False,
                error=str(exc),
            )

    def ingest_directory(
        self, directory: str | Path, recursive: bool = True
    ) -> List[IngestionResult]:
        """Batch-ingest all supported documents in a directory."""
        dir_path = Path(directory)
        results: List[IngestionResult] = []

        for file_path in dir_path.rglob("*") if recursive else dir_path.glob("*"):
            if file_path.suffix.lower() in [".pdf", ".txt", ".md", ".docx"]:
                result = self.ingest_file(file_path)
                results.append(result)

        total_added = sum(r.chunks_added for r in results)
        logger.info(
            f"Batch ingestion complete: {len(results)} files, {total_added} chunks",
        )
        return results

    def _log_to_mlflow(self, result: IngestionResult) -> None:
        """Log ingestion metrics to MLflow for experiment tracking."""
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"ingest_{result.filename}"):
                mlflow.log_params({
                    "strategy": result.strategy,
                    "chunk_size": settings.CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP,
                    "embedding_provider": settings.EMBEDDING_PROVIDER.value,
                })
                mlflow.log_metrics({
                    "chunks_created": result.chunks_created,
                    "chunks_added": result.chunks_added,
                    "total_chars": result.total_chars,
                    "elapsed_seconds": result.elapsed_seconds,
                })
        except Exception as exc:
            # MLflow failures are non-fatal — log and continue
            logger.debug(f"MLflow logging skipped: {exc}")
