"""Per-stage tracing + latency (local stand-in for Cloud Trace / OpenTelemetry).

Each request gets a `Trace` with one `Span` per pipeline stage (guardrails, condense, retrieve,
generate, validate). Spans capture latency_ms + arbitrary attributes (metadata). The trace is:
  * returned in the API response (per-component latency visibility), and
  * fed to metrics + lineage.

GCP mapping: this is exactly what OpenTelemetry -> Cloud Trace does. We keep an in-process
implementation so it works offline; `export_otel()` shows the one-line bridge for Phase 3.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    start: float
    end: float | None = None
    attributes: dict = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return round(((self.end or time.perf_counter()) - self.start) * 1000, 2)

    def set(self, **attrs) -> None:
        self.attributes.update(attrs)


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    spans: list[Span] = field(default_factory=list)
    start: float = field(default_factory=time.perf_counter)

    @contextmanager
    def span(self, name: str, **attrs):
        sp = Span(name=name, start=time.perf_counter(), attributes=dict(attrs))
        self.spans.append(sp)
        try:
            yield sp
        finally:
            sp.end = time.perf_counter()

    @property
    def total_latency_ms(self) -> float:
        return round((time.perf_counter() - self.start) * 1000, 2)

    def latencies(self) -> dict[str, float]:
        """stage -> latency_ms, for API/debug surfaces."""
        return {sp.name: sp.latency_ms for sp in self.spans}

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "total_latency_ms": self.total_latency_ms,
            "spans": [
                {"name": sp.name, "latency_ms": sp.latency_ms, "attributes": sp.attributes}
                for sp in self.spans
            ],
        }


def export_otel(trace: Trace) -> None:
    """Phase-3 bridge: mirror spans to OpenTelemetry (Cloud Trace ingests OTLP).

    Left as a documented no-op locally to avoid exporter noise; enable in the cloud image:
        from opentelemetry import trace as ot
        tracer = ot.get_tracer("enterprise-rag")
        for sp in trace.spans:
            with tracer.start_as_current_span(sp.name) as s:
                for k, v in sp.attributes.items():
                    s.set_attribute(k, v)
    """
    return None
