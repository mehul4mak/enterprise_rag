"""Visualize the document as a page-level graph (Phase 5).

Produces two artifacts from the page graph:
  * an interactive Plotly HTML (self-contained), and
  * a static Matplotlib PNG.

Node = page (size ∝ chunk count). Edge = cross-page relation (width ∝ summed weight).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402


def _layout(pg: nx.Graph):
    return nx.spring_layout(pg, seed=42, weight="weight", k=0.6)


def render_png(pg: nx.Graph, out_path: str) -> str:
    pos = _layout(pg)
    counts = [pg.nodes[n]["chunk_count"] for n in pg.nodes]
    widths = [0.5 + pg[u][v]["weight"] * 0.15 for u, v in pg.edges]

    plt.figure(figsize=(12, 9))
    nx.draw_networkx_edges(pg, pos, width=widths, edge_color="#9db4c0", alpha=0.6)
    nx.draw_networkx_nodes(
        pg, pos, node_size=[120 + c * 90 for c in counts], node_color=counts, cmap="viridis"
    )
    nx.draw_networkx_labels(pg, pos, font_size=8, font_color="white")
    plt.title("Document page graph (node=page, size=chunks, edge=cross-page relation)")
    plt.axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    return out_path


def render_html(pg: nx.Graph, out_path: str) -> str:
    import plotly.graph_objects as go

    pos = _layout(pg)
    edge_x, edge_y = [], []
    for u, v in pg.edges:
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=1, color="#9db4c0"), hoverinfo="none", mode="lines"
    )

    node_x = [pos[n][0] for n in pg.nodes]
    node_y = [pos[n][1] for n in pg.nodes]
    counts = [pg.nodes[n]["chunk_count"] for n in pg.nodes]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[f"p{n}" for n in pg.nodes],
        textposition="middle center",
        hovertext=[
            f"page {n} · {c} chunks · deg {pg.degree(n)}"
            for n, c in zip(pg.nodes, counts, strict=True)
        ],
        marker=dict(
            size=[14 + c * 6 for c in counts],
            color=counts,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="chunks"),
        ),
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Document page graph — hover a page for details",
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="inline")
    return out_path
