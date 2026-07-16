"""Retrieval benchmark — score retrieval configs with NO LLM calls.

Metrics (per question, over the gold set):
  * hit@k  — did any gold page appear in the top-k retrieved chunks?
  * MRR    — 1/rank of the first gold page (0 if none in top-k).

Used to answer, with evidence rather than intuition:
  - which chunk_size / overlap actually retrieves best?
  - which embedding model?
  - do dense / sparse / hybrid / rerank each earn their place?

Usage:
    python -m src.experiments.bench chunking
    python -m src.experiments.bench embeddings
    python -m src.experiments.bench strategies
"""

from __future__ import annotations

import sys
from dataclasses import replace

from ..config import CONFIG, Config
from ..index_store import get_or_build_from_chunks
from ..ingest import ingest_pdf
from ..retriever import _dense_search, _get_reranker, _reciprocal_rank_fusion, _sparse_search
from .goldset import GOLD

PDF = "data/earnings_presentation_q2fy26.pdf"

# Embedding models that need a query prefix to perform correctly (fairness matters in a bench).
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-small-v2": "query: ",
}
DOC_PREFIX = {"intfloat/e5-small-v2": "passage: "}


def _pages_for(idxs: list[int], index) -> list[int]:
    return [index.chunks[i].page for i in idxs]


def _score(ranked_pages: list[int], gold: list[int], k: int) -> tuple[float, float]:
    """Return (hit@k, reciprocal_rank)."""
    top = ranked_pages[:k]
    hit = 1.0 if any(p in gold for p in top) else 0.0
    rr = 0.0
    for rank, p in enumerate(top, 1):
        if p in gold:
            rr = 1.0 / rank
            break
    return hit, rr


def build(cfg: Config):
    chunks = ingest_pdf(PDF, cfg.chunk_size_chars, cfg.chunk_overlap_chars)
    doc_prefix = DOC_PREFIX.get(cfg.embedding_model)
    if doc_prefix:  # e5 needs passages prefixed at index time
        chunks = [replace(c, text=doc_prefix + c.text) for c in chunks]
    return chunks, get_or_build_from_chunks(chunks, cfg)


def rank_pages(q: str, index, cfg: Config, strategy: str, k: int = 5) -> list[int]:
    """Return the retrieved pages in rank order for a strategy."""
    query = QUERY_PREFIX.get(cfg.embedding_model, "") + q

    if strategy == "dense":
        return _pages_for([i for i, _ in _dense_search(query, index, k, cfg)], index)
    if strategy == "sparse":
        return _pages_for([i for i, _ in _sparse_search(q, index, k)], index)

    dense = _dense_search(query, index, cfg.top_k_dense, cfg)
    sparse = _sparse_search(q, index, cfg.top_k_sparse)
    fused = _reciprocal_rank_fusion(dense, sparse)
    order = [i for i, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]
    if strategy == "hybrid":
        return _pages_for(order[:k], index)
    if strategy == "hybrid+rerank":
        rr = _get_reranker(cfg.reranker_model)
        pairs = [(q, index.chunks[i].text) for i in order]
        scores = rr.predict(pairs)
        reranked = [i for i, _ in sorted(zip(order, scores, strict=True), key=lambda t: -t[1])]
        return _pages_for(reranked[:k], index)
    raise ValueError(strategy)


def evaluate(cfg: Config, strategy: str, k: int = 5) -> dict:
    _, index = build(cfg)
    hits, rrs = [], []
    for row in GOLD:
        pages = rank_pages(row["q"], index, cfg, strategy, k)
        h, rr = _score(pages, row["gold_pages"], k)
        hits.append(h)
        rrs.append(rr)
    return {
        "hit@k": round(sum(hits) / len(hits), 3),
        "MRR": round(sum(rrs) / len(rrs), 3),
        "n": len(GOLD),
    }


# ------------------------------------------------------------------ sweeps
def sweep_chunking():
    print(f"{'chunk_size':>10} {'overlap':>8} {'chunks':>7} {'hit@5':>7} {'MRR':>7}")
    for size in (400, 700, 1000, 1500, 2000):
        for ov in (0, 150, 300):
            if ov >= size:
                continue
            cfg = replace(CONFIG, chunk_size_chars=size, chunk_overlap_chars=ov)
            chunks, _ = build(cfg)
            m = evaluate(cfg, "hybrid+rerank")
            print(f"{size:>10} {ov:>8} {len(chunks):>7} {m['hit@k']:>7} {m['MRR']:>7}")


def sweep_embeddings():
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "intfloat/e5-small-v2",
    ]
    print(f"{'embedding model':>40} {'hit@5':>7} {'MRR':>7}")
    for m in models:
        try:
            cfg = replace(CONFIG, embedding_model=m)
            r = evaluate(cfg, "hybrid+rerank")
            print(f"{m:>40} {r['hit@k']:>7} {r['MRR']:>7}")
        except Exception as e:  # noqa: BLE001
            print(f"{m:>40}   FAILED: {type(e).__name__}: {str(e)[:40]}")


def sweep_strategies():
    print(f"{'strategy':>16} {'hit@5':>7} {'MRR':>7}")
    for s in ("sparse", "dense", "hybrid", "hybrid+rerank"):
        r = evaluate(CONFIG, s)
        print(f"{s:>16} {r['hit@k']:>7} {r['MRR']:>7}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "strategies"
    {"chunking": sweep_chunking, "embeddings": sweep_embeddings, "strategies": sweep_strategies}[
        which
    ]()
