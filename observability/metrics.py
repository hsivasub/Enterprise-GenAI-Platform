"""
Metrics & Cost Tracking — Prometheus-compatible
================================================
Tracks request counts, token usage, costs, latency histograms, and
cache hit rates. All metrics are Prometheus-compatible so they work
with standard Grafana dashboards out of the box.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Generator, Optional

from config.settings import settings
from observability.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# Cost model — updated periodically to reflect API pricing
# ─────────────────────────────────────────────────────────────

MODEL_COST_TABLE: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},       # per 1K tokens
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a model call."""
    prices = MODEL_COST_TABLE.get(model)
    if not prices:
        # Fallback to settings-based pricing
        return (
            input_tokens * settings.COST_PER_1K_INPUT_TOKENS / 1000
            + output_tokens * settings.COST_PER_1K_OUTPUT_TOKENS / 1000
        )
    return (
        input_tokens * prices["input"] / 1000
        + output_tokens * prices["output"] / 1000
    )


# ─────────────────────────────────────────────────────────────
# In-memory metrics store (replace with Prometheus client in prod)
# ─────────────────────────────────────────────────────────────

@dataclass
class MetricSnapshot:
    """Aggregated platform metrics at a point in time."""
    total_requests: int = 0
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_retrieval_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    _latency_sum: float = field(default=0.0, repr=False)
    _latency_count: int = field(default=0, repr=False)
    snapshot_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MetricsCollector:
    """
    Thread-safe in-memory metrics aggregator.

    In production, swap the internal store for a Prometheus
    CollectorRegistry + push_to_gateway call. The interface stays identical.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._metrics = MetricSnapshot()
        self._start_time = time.time()

    # ── Recording helpers ──────────────────────────────────

    def record_request(self, latency_ms: float, success: bool = True) -> None:
        with self._lock:
            self._metrics.total_requests += 1
            self._metrics._latency_sum += latency_ms
            self._metrics._latency_count += 1
            self._metrics.avg_latency_ms = (
                self._metrics._latency_sum / self._metrics._latency_count
            )
            if not success:
                self._metrics.errors += 1

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record an LLM call and return the estimated cost."""
        cost = estimate_cost(model, input_tokens, output_tokens)
        with self._lock:
            self._metrics.total_llm_calls += 1
            self._metrics.total_input_tokens += input_tokens
            self._metrics.total_output_tokens += output_tokens
            self._metrics.total_cost_usd += cost
        self._check_budget_alert()
        return cost

    def record_retrieval(self) -> None:
        with self._lock:
            self._metrics.total_retrieval_calls += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self._metrics.cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self._metrics.cache_misses += 1

    def get_snapshot(self) -> MetricSnapshot:
        with self._lock:
            snap = MetricSnapshot(**{
                k: v for k, v in self._metrics.__dict__.items()
            })
            snap.snapshot_time = datetime.now(timezone.utc).isoformat()
            return snap

    def get_prometheus_text(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        snap = self.get_snapshot()
        uptime = time.time() - self._start_time
        lines = [
            "# HELP genai_requests_total Total HTTP requests",
            "# TYPE genai_requests_total counter",
            f"genai_requests_total {snap.total_requests}",
            "",
            "# HELP genai_llm_calls_total Total LLM API calls",
            "# TYPE genai_llm_calls_total counter",
            f"genai_llm_calls_total {snap.total_llm_calls}",
            "",
            "# HELP genai_tokens_total Total tokens processed",
            "# TYPE genai_tokens_total counter",
            f'genai_tokens_total{{type="input"}} {snap.total_input_tokens}',
            f'genai_tokens_total{{type="output"}} {snap.total_output_tokens}',
            "",
            "# HELP genai_cost_usd_total Estimated total LLM cost in USD",
            "# TYPE genai_cost_usd_total counter",
            f"genai_cost_usd_total {snap.total_cost_usd:.6f}",
            "",
            "# HELP genai_cache_hits_total Cache hit count",
            "# TYPE genai_cache_hits_total counter",
            f"genai_cache_hits_total {snap.cache_hits}",
            "",
            "# HELP genai_errors_total Total errors",
            "# TYPE genai_errors_total counter",
            f"genai_errors_total {snap.errors}",
            "",
            "# HELP genai_avg_latency_ms Average request latency (ms)",
            "# TYPE genai_avg_latency_ms gauge",
            f"genai_avg_latency_ms {snap.avg_latency_ms:.2f}",
            "",
            "# HELP genai_uptime_seconds Platform uptime in seconds",
            "# TYPE genai_uptime_seconds gauge",
            f"genai_uptime_seconds {uptime:.0f}",
        ]
        return "\n".join(lines)

    def _check_budget_alert(self) -> None:
        """Log a warning if monthly cost exceeds configured budget."""
        with self._lock:
            cost = self._metrics.total_cost_usd
        if cost > settings.MONTHLY_BUDGET_USD * 0.8:
            logger.warning(
                "Budget alert: 80%% of monthly LLM budget consumed",
                extra={
                    "current_cost_usd": cost,
                    "monthly_budget_usd": settings.MONTHLY_BUDGET_USD,
                    "utilization_pct": round(cost / settings.MONTHLY_BUDGET_USD * 100, 1),
                },
            )


@contextmanager
def track_latency(collector: MetricsCollector) -> Generator[None, None, None]:
    """Context manager that records request latency."""
    start = time.perf_counter()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        collector.record_request(latency_ms, success)


# Global singleton — import from here
metrics = MetricsCollector()
