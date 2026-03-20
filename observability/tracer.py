"""
Distributed Tracing — OpenTelemetry compatible
===============================================
Propagates trace context across service boundaries.
Exports to Jaeger (HTTP), console, or OTLP collector.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional

from observability.logger import get_correlation_id, get_logger

logger = get_logger(__name__)


@dataclass
class Span:
    """Lightweight span compatible with OpenTelemetry data model."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    service: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "OK"
    error_message: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_error(self, message: str) -> None:
        self.status = "ERROR"
        self.error_message = message

    def finish(self) -> None:
        self.end_time = time.perf_counter()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation": self.operation,
            "service": self.service,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "error": self.error_message,
            "attributes": self.attributes,
        }


class Tracer:
    """
    Simple tracer that logs finished spans as structured log events.
    
    In production, replace _export_span with an OTLP exporter:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    """

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name
        self._active_trace_id: Optional[str] = None

    @contextmanager
    def start_span(
        self,
        operation: str,
        parent_span: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Create a new span as a context manager."""
        trace_id = (
            parent_span.trace_id if parent_span else get_correlation_id() or str(uuid.uuid4())
        )
        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span.span_id if parent_span else None,
            operation=operation,
            service=self.service_name,
            attributes=attributes or {},
        )
        try:
            yield span
        except Exception as exc:
            span.set_error(str(exc))
            raise
        finally:
            span.finish()
            self._export_span(span)

    def _export_span(self, span: Span) -> None:
        """Export finished span — currently logs it; swap for OTLP in prod."""
        logger.debug(
            f"SPAN {span.operation} [{span.duration_ms:.1f}ms] {span.status}",
            extra={"event_type": "trace_span", **span.to_dict()},
        )


# One tracer per service — import in service modules
def create_tracer(service_name: str) -> Tracer:
    return Tracer(service_name)
