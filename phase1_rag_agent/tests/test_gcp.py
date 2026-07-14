"""Phase 3 GCP provider tests — construction + config guards (no GCP SDKs / creds needed).

These verify the wiring: BACKEND=gcp selects the Vertex/Document AI providers, they construct
cheaply (lazy imports), and a missing config raises a clear ValueError *before* any SDK/auth call.
"""

import pytest

from src.config import Config
from src.providers.factory import (
    build_document_parser,
    build_embedder,
    build_llm,
    build_retriever,
)
from src.providers.gcp import (
    DocumentAIParser,
    VertexEmbedder,
    VertexGeminiLLM,
    VertexVectorRetriever,
)


def _gcp_cfg(**kw):
    return Config(backend="gcp", **kw)


def test_gcp_factory_selects_vertex_impls():
    cfg = _gcp_cfg()
    assert isinstance(build_document_parser(cfg), DocumentAIParser)
    assert isinstance(build_embedder(cfg), VertexEmbedder)
    assert isinstance(build_retriever(cfg), VertexVectorRetriever)
    assert isinstance(build_llm(cfg), VertexGeminiLLM)


def test_missing_project_raises_clear_error_not_auth_crash():
    cfg = _gcp_cfg(gcp_project="")
    with pytest.raises(ValueError, match="gcp_project"):
        VertexGeminiLLM(cfg).generate("hi", "sys")
    with pytest.raises(ValueError, match="gcp_project"):
        VertexEmbedder(cfg).embed(["hi"])
    with pytest.raises(ValueError, match="docai_processor_id|gcp_project"):
        DocumentAIParser(cfg).extract_pages("x.pdf")


def test_retriever_reports_empty_before_indexing():
    r = VertexVectorRetriever(_gcp_cfg(gcp_project="p"))
    assert r.num_chunks == 0 and r.all_chunks() == []


def test_default_region_is_india_sovereign():
    assert Config().gcp_location == "asia-south1"
