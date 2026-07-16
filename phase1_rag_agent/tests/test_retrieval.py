"""Structured (metadata) retrieval unit tests — no models needed.

structured_lookup only reads chunk.page, so we can test it against a lightweight stand-in index.
"""

from types import SimpleNamespace

from src.ingest import Chunk
from src.retriever import structured_lookup


def _index(pages):
    chunks = [Chunk(chunk_id=f"c{i}", page=p, text=f"text {i}") for i, p in enumerate(pages, 1)]
    return SimpleNamespace(chunks=chunks)


def test_detects_page_reference():
    idx = _index([1, 1, 2, 3])
    assert structured_lookup("what is on page 1", idx) == [0, 1]
    assert structured_lookup("summarize slide 3", idx) == [3]
    assert structured_lookup("show me pg. 2", idx) == [2]


def test_no_positional_reference_returns_empty():
    idx = _index([1, 2, 3])
    assert structured_lookup("what is the total income", idx) == []
    # a bare number (not preceded by page/slide/pg) must NOT trigger
    assert structured_lookup("revenue in 2024", idx) == []


def test_missing_page_returns_empty():
    assert structured_lookup("open page 9", _index([1, 2])) == []
