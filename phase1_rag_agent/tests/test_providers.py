"""Phase 2 tests: factory selection, GCP stub guards, and graph wiring (no live LLM)."""

import pytest

from src.config import CONFIG, Config
from src.providers.base import DocumentParser, Embedder, LLMProvider, Retriever
from src.providers.factory import (
    build_document_parser,
    build_embedder,
    build_llm,
    build_retriever,
)


def test_local_factory_returns_local_impls():
    parser = build_document_parser(CONFIG)
    assert isinstance(parser, DocumentParser)
    assert isinstance(build_embedder(CONFIG), Embedder)
    assert isinstance(build_retriever(CONFIG), Retriever)
    assert isinstance(build_llm(CONFIG), LLMProvider)


def test_unknown_backend_raises():
    bad = Config(backend="azure")
    with pytest.raises(ValueError):
        build_llm(bad)


def test_gcp_backend_selects_stubs_that_raise_not_implemented():
    gcp = Config(backend="gcp")
    parser = build_document_parser(gcp)
    # Stub is constructed fine but refuses to run until Phase 3 wires it.
    with pytest.raises(NotImplementedError):
        parser.extract_pages("whatever.pdf")
    with pytest.raises(NotImplementedError):
        build_llm(gcp).generate("hi", "sys")


def test_graph_compiles_over_stub_providers():
    """The LangGraph topology builds without any LLM/retriever calls."""
    from src.graph import build_graph
    from src.providers.local import LocalHybridRetriever, LocalLLM

    graph = build_graph(LocalHybridRetriever(CONFIG), LocalLLM(CONFIG), CONFIG)
    assert graph is not None
