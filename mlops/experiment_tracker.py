"""
MLflow Experiment Tracker — MLOps Layer
=========================================
Tracks all platform experiments, model versions, and pipeline runs.
Provides a unified interface for logging parameters, metrics, and artifacts.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from config.settings import settings
from observability.logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """
    MLflow-backed experiment tracker.
    Falls back to no-op logging if MLflow is unavailable.
    """

    def __init__(
        self,
        tracking_uri: str = settings.MLFLOW_TRACKING_URI,
        experiment_name: str = settings.MLFLOW_EXPERIMENT_NAME,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._available = self._init_mlflow()

    def _init_mlflow(self) -> bool:
        try:
            import mlflow
            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment_name)
            logger.info(
                f"MLflow connected: {self._tracking_uri} / {self._experiment_name}"
            )
            return True
        except Exception as exc:
            logger.warning(f"MLflow unavailable: {exc}. Experiment tracking disabled.")
            return False

    @contextmanager
    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Generator:
        """Context manager for an MLflow run."""
        if not self._available:
            yield _NoOpRun()
            return

        import mlflow
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run

    def log_rag_evaluation(
        self,
        *,
        question: str,
        answer: str,
        context: str,
        metrics_dict: Dict[str, float],
        run_name: str = "rag_evaluation",
    ) -> None:
        """Log a RAG evaluation result."""
        if not self._available:
            logger.debug(f"[MLflow disabled] RAG eval metrics: {metrics_dict}")
            return

        import mlflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "question_length": len(question),
                "answer_length": len(answer),
                "context_length": len(context),
                "llm_model": settings.OPENAI_MODEL,
                "embedding_model": settings.EMBEDDING_MODEL,
                "chunk_size": settings.CHUNK_SIZE,
            })
            mlflow.log_metrics(metrics_dict)
            mlflow.log_text(question, "question.txt")
            mlflow.log_text(answer, "answer.txt")
            mlflow.log_text(context, "context.txt")

    def log_ingestion_run(
        self,
        *,
        filename: str,
        strategy: str,
        chunks: int,
        elapsed_s: float,
    ) -> None:
        if not self._available:
            return

        import mlflow
        with self.start_run(f"ingest_{filename}"):
            mlflow.log_params({"strategy": strategy, "model": settings.EMBEDDING_MODEL})
            mlflow.log_metrics({"chunks": chunks, "elapsed_seconds": elapsed_s})

    def register_model(
        self,
        model_name: str,
        run_id: str,
        artifact_path: str = "model",
    ) -> None:
        """Register a model version in the MLflow Model Registry."""
        if not self._available:
            return

        import mlflow
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mlflow.register_model(model_uri=model_uri, name=model_name)
        logger.info(f"Model registered: {model_name} from run {run_id}")


class _NoOpRun:
    """Dummy run object when MLflow is unavailable."""
    def log_metric(self, *args, **kwargs): pass
    def log_param(self, *args, **kwargs): pass
    def log_artifact(self, *args, **kwargs): pass


# ─────────────────────────────────────────────────────────────
# Pipeline Config Management
# ─────────────────────────────────────────────────────────────

class PipelineConfig:
    """
    Config-driven pipeline parameters.
    Enables zero-code changes for A/B tests.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or self._defaults()

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        return {
            "ingestion": {
                "chunking_strategy": "recursive_character",
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "embedding_provider": settings.EMBEDDING_PROVIDER.value,
            },
            "retrieval": {
                "top_k": settings.FAISS_TOP_K,
                "min_score": 0.3,
                "enable_reranking": settings.ENABLE_RERANKING,
                "enable_hybrid": settings.ENABLE_HYBRID_SEARCH,
            },
            "llm": {
                "provider": settings.LLM_PROVIDER.value,
                "model": settings.OPENAI_MODEL,
                "temperature": settings.OPENAI_TEMPERATURE,
                "max_tokens": settings.OPENAI_MAX_TOKENS,
            },
            "agent": {
                "max_iterations": settings.AGENT_MAX_ITERATIONS,
                "timeout_seconds": settings.AGENT_TIMEOUT_SECONDS,
            },
            "guardrails": {
                "enable_pii_detection": settings.ENABLE_PII_DETECTION,
                "enable_content_filter": settings.ENABLE_CONTENT_FILTER,
                "max_input_length": settings.MAX_INPUT_LENGTH,
            },
        }

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._config.get(section, {}).get(key, default)

    def to_mlflow_params(self) -> Dict[str, str]:
        """Flatten config for MLflow parameter logging."""
        flat = {}
        for section, values in self._config.items():
            for k, v in values.items():
                flat[f"{section}.{k}"] = str(v)
        return flat


# Singletons
tracker = ExperimentTracker()
pipeline_config = PipelineConfig()
