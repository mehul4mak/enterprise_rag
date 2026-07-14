"""Semantic answer cache — the marquee latency/cost optimization.

A repeated or near-duplicate question should not re-run retrieval + generation. We embed the query
(cheap, local) and, if a previously answered query is within a cosine threshold, return the cached
answer — skipping the LLM entirely (zero cost, ~ms latency).

Scope & correctness:
  * Keyed per **document** (source path) so answers never leak across documents.
  * Only first-turn (no chat history) questions are cached — a follow-up like "break that down"
    depends on history and must not be served from a query-only cache.

GCP mapping: a managed semantic cache on Memorystore/Vertex; the pattern is identical.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class CacheEntry:
    embedding: np.ndarray
    answer: str
    retrieved: list  # list[RetrievedChunk]


class SemanticCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_doc: dict[str, list[CacheEntry]] = {}

    def get(self, doc_id: str, embedding: np.ndarray, threshold: float):
        """Return a cached (answer, retrieved) for a near-duplicate query, or None."""
        with self._lock:
            entries = self._by_doc.get(doc_id, [])
            best, best_sim = None, -1.0
            for e in entries:
                sim = float(np.dot(e.embedding, embedding))  # normalized → cosine
                if sim > best_sim:
                    best, best_sim = e, sim
            if best is not None and best_sim >= threshold:
                return best.answer, best.retrieved
        return None

    def put(self, doc_id: str, embedding: np.ndarray, answer: str, retrieved: list) -> None:
        with self._lock:
            self._by_doc.setdefault(doc_id, []).append(
                CacheEntry(embedding=embedding, answer=answer, retrieved=retrieved)
            )

    def clear(self) -> None:
        with self._lock:
            self._by_doc.clear()

    def size(self, doc_id: str | None = None) -> int:
        with self._lock:
            if doc_id is not None:
                return len(self._by_doc.get(doc_id, []))
            return sum(len(v) for v in self._by_doc.values())


# Process-wide cache (a real deployment scopes per tenant / uses Memorystore).
CACHE = SemanticCache()
