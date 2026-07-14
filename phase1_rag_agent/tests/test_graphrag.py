"""GraphRAG unit tests — graph construction + retriever interface (deterministic, no LLM)."""

from src.config import CONFIG
from src.graphrag.graph_build import build_doc_graph, page_graph
from src.graphrag.retriever import GraphRetriever
from src.ingest import Chunk
from src.providers.base import Retriever


def _chunks():
    return [
        Chunk(
            chunk_id="c1",
            page=1,
            text="Adani Enterprises total income grew in H1-26 to 44,281 crore.",
        ),
        Chunk(
            chunk_id="c2", page=1, text="Total income and EBITDA highlights for the H1-26 period."
        ),
        Chunk(chunk_id="c3", page=2, text="Airport passenger traffic and cargo volume increased."),
        Chunk(chunk_id="c4", page=2, text="Cargo tonnage and passenger counts at airports rose."),
        Chunk(
            chunk_id="c5", page=3, text="Green hydrogen and solar manufacturing capacity expansion."
        ),
    ]


def test_graph_builds_nodes_and_edges():
    doc = build_doc_graph(_chunks(), CONFIG)
    g = doc.graph
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() > 0
    # same-page chunks must be adjacency-linked
    assert g.has_edge("c1", "c2")
    assert "adjacency" in g["c1"]["c2"]["relations"]


def test_edges_carry_weight_and_relations():
    doc = build_doc_graph(_chunks(), CONFIG)
    for _, _, data in doc.graph.edges(data=True):
        assert data["weight"] > 0
        assert data["relations"]


def test_page_graph_aggregates_to_pages():
    doc = build_doc_graph(_chunks(), CONFIG)
    pg = page_graph(doc)
    assert set(pg.nodes) == {1, 2, 3}
    assert pg.nodes[1]["chunk_count"] == 2


def test_graph_retriever_conforms_and_ranks():
    r = GraphRetriever(CONFIG)
    assert isinstance(r, Retriever)
    r.index(_chunks())
    assert r.num_chunks == 5
    assert len(r.all_chunks()) == 5

    res = r.retrieve("airport cargo and passenger volume", k=3)
    assert len(res) == 3
    # a relevant airport chunk should surface in the top-3
    cites = [x.chunk.chunk_id for x in res]
    assert "c3" in cites or "c4" in cites
    # scores are populated
    assert all(x.fused_score is not None for x in res)


def test_retrieve_before_index_raises():
    import pytest

    with pytest.raises(RuntimeError):
        GraphRetriever(CONFIG).retrieve("q", 3)
