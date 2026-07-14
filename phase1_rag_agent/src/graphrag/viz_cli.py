"""Regenerate the page-graph visualization from a PDF.

python -m src.graphrag.viz_cli --pdf data/earnings_presentation_q2fy26.pdf
"""

import argparse

from ..config import CONFIG
from ..ingest import ingest_pdf
from .graph_build import build_doc_graph, page_graph
from .visualize import render_html, render_png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out-dir", default="docs/viz")
    args = ap.parse_args()

    chunks = ingest_pdf(args.pdf, CONFIG.chunk_size_chars, CONFIG.chunk_overlap_chars)
    doc = build_doc_graph(chunks, CONFIG)
    pg = page_graph(doc)
    png = render_png(pg, f"{args.out_dir}/page_graph.png")
    html = render_html(pg, f"{args.out_dir}/page_graph.html")
    print(
        f"chunk graph: {doc.graph.number_of_nodes()} nodes / {doc.graph.number_of_edges()} edges | "
        f"page graph: {pg.number_of_nodes()} pages / {pg.number_of_edges()} edges"
    )
    print(f"wrote {png} and {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
