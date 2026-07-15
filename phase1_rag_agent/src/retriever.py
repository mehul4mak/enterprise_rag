"""Hybrid retrieval: dense (FAISS) + sparse (BM25) fused via Reciprocal Rank Fusion,
then refined with a cross-encoder reranker — plus a structured (metadata) lookup so positional
questions like "what is on page 3" work, which neither text retriever can serve (the page number
is metadata, not a token in the chunk text)."""

import re
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from .config import Config
from .embeddings import embed_texts
from .index_store import HybridIndex, tokenize
from .ingest import Chunk

_reranker_cache: dict[str, CrossEncoder] = {}

# Positional references: "page 3", "on page 12", "slide 5", "pg. 7".
_PAGE_RE = re.compile(r"\b(?:page|slide|pg)\s*\.?\s*(\d+)\b")


def _get_reranker(model_name: str) -> CrossEncoder:
    if model_name not in _reranker_cache:
        _reranker_cache[model_name] = CrossEncoder(model_name)
    return _reranker_cache[model_name]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    dense_score: float | None
    sparse_score: float | None
    fused_score: float
    rerank_score: float | None = None

    @property
    def display_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fused_score


def _dense_search(
    query: str, index: HybridIndex, k: int, config: Config
) -> list[tuple[int, float]]:
    qvec = embed_texts([query], config.embedding_model)
    k = min(k, index.faiss_index.ntotal)
    scores, ids = index.faiss_index.search(qvec, k)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=False) if i != -1]


def _sparse_search(query: str, index: HybridIndex, k: int) -> list[tuple[int, float]]:
    scores = index.bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(i, float(scores[i])) for i in ranked]


def _reciprocal_rank_fusion(
    dense: list[tuple[int, float]], sparse: list[tuple[int, float]], rrf_k: int = 60
) -> dict[int, float]:
    fused: dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, (idx, _) in enumerate(sparse):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return fused


def structured_lookup(query: str, index: HybridIndex) -> list[int]:
    """Detect a positional reference (e.g. "page 3") and return that page's chunk indices.

    This is *structured* retrieval: it queries the `page` metadata field that FAISS (meaning) and
    BM25 (content tokens) can't — because the page number isn't text inside the chunk. Returns [] if
    the question has no positional reference or the page doesn't exist.
    """
    m = _PAGE_RE.search(query.lower())
    if not m:
        return []
    page = int(m.group(1))
    return [i for i, c in enumerate(index.chunks) if c.page == page]


def retrieve(
    query: str, index: HybridIndex, config: Config, use_reranker: bool = True
) -> list[RetrievedChunk]:
    dense = _dense_search(query, index, config.top_k_dense, config)
    sparse = _sparse_search(query, index, config.top_k_sparse)

    dense_scores = dict(dense)
    sparse_scores = dict(sparse)
    fused = _reciprocal_rank_fusion(dense, sparse)

    candidates = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    results = [
        RetrievedChunk(
            chunk=index.chunks[idx],
            dense_score=dense_scores.get(idx),
            sparse_score=sparse_scores.get(idx),
            fused_score=score,
        )
        for idx, score in candidates
    ]

    if use_reranker and results:
        reranker = _get_reranker(config.reranker_model)
        pairs = [(query, r.chunk.text) for r in results]
        rerank_scores = reranker.predict(pairs)
        for r, s in zip(results, rerank_scores, strict=False):
            r.rerank_score = float(s)
        results.sort(key=lambda r: r.rerank_score, reverse=True)

    # Structured retrieval: if the question names a page/slide, pin that page's chunks to the front
    # (guaranteed inclusion — text search can't find them, and the reranker would bury them).
    structured_idxs = structured_lookup(query, index)
    if structured_idxs:
        pinned = [
            RetrievedChunk(
                chunk=index.chunks[i], dense_score=None, sparse_score=None, fused_score=1.0
            )
            for i in structured_idxs
        ]
        pinned_ids = {p.chunk.chunk_id for p in pinned}
        results = pinned + [r for r in results if r.chunk.chunk_id not in pinned_ids]

    return results[: config.top_k_final]
