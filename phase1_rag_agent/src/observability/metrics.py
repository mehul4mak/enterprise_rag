"""In-process metrics registry (local stand-in for Cloud Monitoring).

Tracks, per pipeline stage:
  * a call counter,
  * a latency histogram (count/sum/min/max/p50/p95), and
  * arbitrary named counters (e.g. guardrail blocks, refusals).

GCP mapping: Cloud Monitoring custom metrics. `snapshot()` is what a /metrics endpoint or a
Monitoring exporter would publish.
"""

from __future__ import annotations

import threading
from bisect import insort
from dataclasses import dataclass, field


@dataclass
class _Latencies:
    values: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        insort(self.values, ms)

    def _pct(self, p: float) -> float:
        if not self.values:
            return 0.0
        idx = min(len(self.values) - 1, int(round(p / 100 * (len(self.values) - 1))))
        return round(self.values[idx], 2)

    def summary(self) -> dict:
        v = self.values
        return {
            "count": len(v),
            "sum_ms": round(sum(v), 2),
            "min_ms": round(v[0], 2) if v else 0.0,
            "max_ms": round(v[-1], 2) if v else 0.0,
            "p50_ms": self._pct(50),
            "p95_ms": self._pct(95),
        }


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stage_latency: dict[str, _Latencies] = {}
        self._counters: dict[str, int] = {}

    def record_stage(self, stage: str, latency_ms: float) -> None:
        with self._lock:
            self._stage_latency.setdefault(stage, _Latencies()).add(latency_ms)
            self._counters[f"stage.{stage}.calls"] = (
                self._counters.get(f"stage.{stage}.calls", 0) + 1
            )

    def incr(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + by

    def record_trace(self, trace) -> None:
        """Ingest every span of a completed trace."""
        for sp in trace.spans:
            self.record_stage(sp.name, sp.latency_ms)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": dict(sorted(self._counters.items())),
                "stage_latency": {
                    stage: lat.summary() for stage, lat in sorted(self._stage_latency.items())
                },
            }


# Process-wide singleton (a real deployment would scope per-tenant).
METRICS = Metrics()
