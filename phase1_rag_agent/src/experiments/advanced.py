"""HyDE and Contextual Retrieval — two advanced retrieval techniques, benchmarked honestly.

**HyDE** (Hypothetical Document Embeddings): instead of embedding the *question* (which looks
nothing like the answer text), ask an LLM to write a *hypothetical answer*, then embed THAT and
search with it. Rationale: a fake answer lives in the same "shape" of language as the real passage,
so it should be closer in embedding space than the question is. Cost: +1 LLM call per query.

**Contextual Retrieval** (Anthropic's technique): each chunk is embedded/indexed with a short blurb
describing where it sits in the document, so an isolated chunk of numbers stops being ambiguous.
  * `contextual-cheap`  — deterministic blurb (doc title + page no.). Free, no LLM.
  * `contextual-llm`    — the full method: an LLM writes the blurb per chunk. Costs one LLM call per
                          chunk (63 for this deck) — implemented, off by default to spare quota.

Both are scored with the same no-LLM metrics as bench.py (hit@k / MRR) on the gold set, so the
comparison against plain hybrid+rerank is apples-to-apples.

Usage:
    python -m src.experiments.advanced contextual    # free (no LLM)
    python -m src.experiments.advanced hyde          # ~6 LLM calls (one per gold question)
"""

from __future__ import annotations

import sys
from dataclasses import replace

from ..config import CONFIG, Config
from ..index_store import get_or_build_from_chunks
from ..ingest import Chunk, ingest_pdf
from ..llm import generate
from ..retriever import _dense_search, _get_reranker, _reciprocal_rank_fusion, _sparse_search
from .bench import PDF, _pages_for, _score
from .goldset import GOLD

DOC_TITLE = "Adani Enterprises Limited (AEL) Q2 FY26 earnings presentation"

HYDE_SYSTEM = (
    "You write a short, plausible passage that would ANSWER the user's question, as if excerpted "
    "from a corporate earnings presentation. Invent realistic figures/labels — it is only used as a "
    "search probe, never shown to a user. 2 sentences max, no preamble."
)

CONTEXT_SYSTEM = (
    "Given a chunk from a financial presentation, write ONE short sentence (max 20 words) situating "
    "it in the document: what section/topic it belongs to. No preamble."
)


def _hyde_query(question: str, config: Config) -> str:
    """Generate a hypothetical answer to use as the search probe."""
    try:
        return generate(f"Question: {question}\n\nPassage:", HYDE_SYSTEM, config)
    except Exception:  # noqa: BLE001 — fall back to the raw question
        return question


def _cheap_context(chunk: Chunk) -> str:
    return f"[Context: {DOC_TITLE}, page {chunk.page}.] "


def _llm_context(chunk: Chunk, config: Config) -> str:
    try:
        blurb = generate(
            f"Document: {DOC_TITLE}\n\nChunk:\n{chunk.text[:800]}\n\nOne-sentence context:",
            CONTEXT_SYSTEM,
            config,
        )
        return f"[Context: {blurb}] "
    except Exception:  # noqa: BLE001
        return _cheap_context(chunk)


def contextualize(chunks: list[Chunk], config: Config, mode: str) -> list[Chunk]:
    """Prepend a situating blurb to each chunk's text before indexing."""
    if mode == "cheap":
        return [replace(c, text=_cheap_context(c) + c.text) for c in chunks]
    if mode == "llm":
        return [replace(c, text=_llm_context(c, config) + c.text) for c in chunks]
    raise ValueError(mode)


def _rank(query_for_dense: str, query_for_sparse: str, index, cfg: Config, k: int) -> list[int]:
    """hybrid+rerank, but allowing a different probe for the dense side (that's what HyDE changes)."""
    dense = _dense_search(query_for_dense, index, cfg.top_k_dense, cfg)
    sparse = _sparse_search(query_for_sparse, index, cfg.top_k_sparse)
    fused = _reciprocal_rank_fusion(dense, sparse)
    order = [i for i, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]
    rr = _get_reranker(cfg.reranker_model)
    scores = rr.predict([(query_for_sparse, index.chunks[i].text) for i in order])
    reranked = [i for i, _ in sorted(zip(order, scores, strict=True), key=lambda t: -t[1])]
    return _pages_for(reranked[:k], index)


def run_contextual(k: int = 5):
    base_chunks = ingest_pdf(PDF, CONFIG.chunk_size_chars, CONFIG.chunk_overlap_chars)
    print(f"{'variant':>20} {'hit@5':>7} {'MRR':>7}")
    for name, chunks in [
        ("baseline", base_chunks),
        ("contextual-cheap", contextualize(base_chunks, CONFIG, "cheap")),
    ]:
        index = get_or_build_from_chunks(chunks, CONFIG)
        hits, rrs = [], []
        for row in GOLD:
            pages = _rank(row["q"], row["q"], index, CONFIG, k)
            h, r = _score(pages, row["gold_pages"], k)
            hits.append(h)
            rrs.append(r)
        print(f"{name:>20} {sum(hits) / len(hits):>7.3f} {sum(rrs) / len(rrs):>7.3f}")


def run_hyde(k: int = 5):
    chunks = ingest_pdf(PDF, CONFIG.chunk_size_chars, CONFIG.chunk_overlap_chars)
    index = get_or_build_from_chunks(chunks, CONFIG)
    print(f"provider={CONFIG.llm_provider} model={CONFIG.active_model}")
    print(f"{'variant':>20} {'hit@5':>7} {'MRR':>7}")
    for name in ("baseline", "hyde"):
        hits, rrs = [], []
        for row in GOLD:
            probe = _hyde_query(row["q"], CONFIG) if name == "hyde" else row["q"]
            # HyDE replaces only the DENSE probe; BM25 still uses the real question (keywords matter).
            pages = _rank(probe, row["q"], index, CONFIG, k)
            h, r = _score(pages, row["gold_pages"], k)
            hits.append(h)
            rrs.append(r)
        print(f"{name:>20} {sum(hits) / len(hits):>7.3f} {sum(rrs) / len(rrs):>7.3f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "contextual"
    {"contextual": run_contextual, "hyde": run_hyde}[which]()
