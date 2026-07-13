"""Deterministic tests for grounding / refusal logic (no LLM calls)."""

from src.agent import NOT_FOUND, RAGAgent
from src.config import CONFIG
from src.ingest import Chunk
from src.retriever import RetrievedChunk


def _fake_retrieved():
    chunks = [
        Chunk(chunk_id="c35", page=22, text="TOTAL INCOME 44,281"),
        Chunk(chunk_id="c2", page=2, text="Total Income 49,263 44,281"),
    ]
    return [
        RetrievedChunk(chunk=c, dense_score=1.0, sparse_score=1.0, fused_score=1.0) for c in chunks
    ]


def _agent():
    # index is unused by _postprocess; pass a stub.
    return RAGAgent(index=None, config=CONFIG)  # type: ignore[arg-type]


def test_valid_citation_passes_through():
    agent = _agent()
    out = agent._postprocess("Total income was 44,281 [p22:c35].", _fake_retrieved())
    assert out == "Total income was 44,281 [p22:c35]."


def test_page_only_citation_is_valid():
    agent = _agent()
    out = agent._postprocess("It was 44,281 [p22].", _fake_retrieved())
    assert "[p22]" in out


def test_no_citation_becomes_refusal():
    agent = _agent()
    out = agent._postprocess("The total income was 44,281.", _fake_retrieved())
    assert out == NOT_FOUND


def test_hallucinated_citation_becomes_refusal():
    # Cites a page/chunk that was never retrieved.
    agent = _agent()
    out = agent._postprocess("The CEO email is x@y.com [p99:c99].", _fake_retrieved())
    assert out == NOT_FOUND


def test_explicit_not_found_is_normalized():
    agent = _agent()
    out = agent._postprocess("Not found in the document. I checked.", _fake_retrieved())
    assert out == NOT_FOUND
