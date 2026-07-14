"""Build a knowledge graph over a document's chunks (Phase 5 GraphRAG).

Nodes = chunks (each tagged with its page). Edges capture three relations, each weighted:
  * adjacency  — consecutive chunks / same-page chunks (document flow)
  * semantic   — cosine similarity between chunk embeddings above a threshold
  * keyword    — Jaccard overlap of significant tokens above a threshold

The graph + embedding matrix feed the GraphRetriever (personalized PageRank) and the page-level
visualization. Everything is local (networkx + sentence-transformers); no cloud.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..config import Config
from ..embeddings import embed_texts
from ..index_store import tokenize
from ..ingest import Chunk


@dataclass
class DocGraph:
    graph: nx.Graph  # nodes keyed by chunk_id
    chunks: list[Chunk]
    embeddings: np.ndarray  # (n, d), L2-normalized, row i == chunks[i]
    id_to_idx: dict[str, int]


def _significant_tokens(text: str, min_len: int = 4) -> set[str]:
    return {t for t in tokenize(text) if len(t) >= min_len}


def build_doc_graph(
    chunks: list[Chunk],
    config: Config,
    sim_threshold: float = 0.45,
    kw_threshold: float = 0.18,
    adjacency_weight: float = 0.6,
) -> DocGraph:
    """Construct the weighted chunk graph. O(n^2) over chunks — fine for tens/hundreds."""
    if not chunks:
        raise ValueError("No chunks to build a graph from.")

    embeddings = embed_texts([c.text for c in chunks], config.embedding_model)
    id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}
    token_sets = [_significant_tokens(c.text) for c in chunks]

    g = nx.Graph()
    for c in chunks:
        g.add_node(c.chunk_id, page=c.page, citation=c.citation, chars=len(c.text))

    # Cosine similarity matrix (embeddings are normalized → dot product == cosine).
    sims = embeddings @ embeddings.T
    n = len(chunks)

    for i in range(n):
        # (a) adjacency: consecutive chunk + same-page neighbours
        for j in (i + 1,):
            if j < n:
                _add_edge(g, chunks[i].chunk_id, chunks[j].chunk_id, adjacency_weight, "adjacency")
        for j in range(i + 1, n):
            if chunks[j].page == chunks[i].page:
                _add_edge(g, chunks[i].chunk_id, chunks[j].chunk_id, adjacency_weight, "adjacency")

            # (b) semantic similarity
            cos = float(sims[i, j])
            if cos >= sim_threshold:
                _add_edge(g, chunks[i].chunk_id, chunks[j].chunk_id, cos, "semantic")

            # (c) keyword overlap (Jaccard)
            a, b = token_sets[i], token_sets[j]
            if a and b:
                jacc = len(a & b) / len(a | b)
                if jacc >= kw_threshold:
                    _add_edge(g, chunks[i].chunk_id, chunks[j].chunk_id, jacc, "keyword")

    return DocGraph(graph=g, chunks=chunks, embeddings=embeddings, id_to_idx=id_to_idx)


def _add_edge(g: nx.Graph, u: str, v: str, weight: float, relation: str) -> None:
    """Add/merge an edge, keeping the max weight and recording contributing relations."""
    if g.has_edge(u, v):
        data = g[u][v]
        data["weight"] = max(data["weight"], weight)
        data["relations"].add(relation)
    else:
        g.add_edge(u, v, weight=weight, relations={relation})


def page_graph(doc: DocGraph) -> nx.Graph:
    """Aggregate the chunk graph to a page-level graph (for visualization).

    Page nodes carry chunk_count; page edges carry summed chunk-edge weight + count.
    """
    pg = nx.Graph()
    for c in doc.chunks:
        if pg.has_node(c.page):
            pg.nodes[c.page]["chunk_count"] += 1
        else:
            pg.add_node(c.page, chunk_count=1)

    for u, v, data in doc.graph.edges(data=True):
        pu = doc.graph.nodes[u]["page"]
        pv = doc.graph.nodes[v]["page"]
        if pu == pv:
            continue
        if pg.has_edge(pu, pv):
            pg[pu][pv]["weight"] += data["weight"]
            pg[pu][pv]["count"] += 1
        else:
            pg.add_edge(pu, pv, weight=data["weight"], count=1)
    return pg
