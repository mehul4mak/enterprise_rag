"""GraphRetriever — graph-based retrieval via personalized PageRank (Phase 5).

Strategy:
  1. Embed the query; score every chunk by cosine similarity (the "seed" signal).
  2. Run **personalized PageRank** on the chunk graph, personalized toward the query-similar seeds,
     so relevance flows across adjacency / semantic / keyword edges (multi-hop).
  3. Blend:  final = alpha * cosine  +  (1 - alpha) * pagerank  (both min-max normalized).
  4. Return the top-k chunks.

Implements the Retriever ABC, so it drops into the LangGraph agent unchanged. Reranking is left to
the blend rather than a cross-encoder (keeps it a pure graph method for the comparison).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from ..config import Config
from ..embeddings import embed_texts
from ..ingest import Chunk
from ..providers.base import Retriever
from ..retriever import RetrievedChunk
from .graph_build import DocGraph, build_doc_graph


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class GraphRetriever(Retriever):
    def __init__(self, config: Config, alpha: float = 0.5, seed_top_n: int = 8):
        self.config = config
        self.alpha = alpha  # weight on direct cosine vs graph pagerank
        self.seed_top_n = seed_top_n
        self._doc: DocGraph | None = None

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks to index.")
        self._doc = build_doc_graph(chunks, self.config)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._doc is None:
            raise RuntimeError("GraphRetriever.index() must be called before retrieve().")
        doc = self._doc
        n = len(doc.chunks)

        qvec = embed_texts([query], self.config.embedding_model)[0]
        cosine = doc.embeddings @ qvec  # (n,), normalized embeddings → cosine

        # Personalization: softmax over the top-N seed similarities, others 0.
        seeds = np.argsort(-cosine)[: min(self.seed_top_n, n)]
        personalization = {c.chunk_id: 0.0 for c in doc.chunks}
        seed_scores = np.exp(cosine[seeds] - cosine[seeds].max())
        seed_scores /= seed_scores.sum()
        for idx, s in zip(seeds, seed_scores, strict=True):
            personalization[doc.chunks[idx].chunk_id] = float(s)

        pr = self._pagerank(doc.graph, personalization)
        pr_vec = np.array([pr.get(c.chunk_id, 0.0) for c in doc.chunks], dtype="float32")

        cos_n = _minmax(cosine)
        pr_n = _minmax(pr_vec)
        final = self.alpha * cos_n + (1.0 - self.alpha) * pr_n

        order = np.argsort(-final)[:k]
        results: list[RetrievedChunk] = []
        for idx in order:
            results.append(
                RetrievedChunk(
                    chunk=doc.chunks[idx],
                    dense_score=float(cosine[idx]),
                    sparse_score=None,
                    fused_score=float(final[idx]),
                    rerank_score=float(pr_vec[idx]),  # expose the pagerank as the "rerank" signal
                )
            )
        return results

    @staticmethod
    def _pagerank(graph: nx.Graph, personalization: dict[str, float]) -> dict[str, float]:
        if graph.number_of_edges() == 0:
            return personalization
        try:
            return nx.pagerank(
                graph, alpha=0.85, personalization=personalization, weight="weight", max_iter=200
            )
        except nx.PowerIterationFailedConvergence:
            return personalization

    @property
    def num_chunks(self) -> int:
        return 0 if self._doc is None else len(self._doc.chunks)

    def all_chunks(self) -> list[Chunk]:
        return [] if self._doc is None else list(self._doc.chunks)
