# Phase 5 — GraphRAG (page/chunk graph + graph retrieval)

An experiment: represent the document as a **graph** and retrieve by walking it, instead of the
Phase 1–4 hybrid (FAISS + BM25 + cross-encoder). Includes a **page-level visualization**.

## Design

**Graph construction** (`src/graphrag/graph_build.py`) — nodes are chunks (each tagged with its
page); edges come from three signals, each weighted:
- **adjacency** — consecutive chunks and same-page chunks (document flow),
- **semantic** — cosine similarity between chunk embeddings ≥ `sim_threshold` (0.45),
- **keyword** — Jaccard overlap of significant tokens ≥ `kw_threshold` (0.18).

**Retrieval** (`src/graphrag/retriever.py`, implements the `Retriever` ABC) —
1. embed the query, score every chunk by cosine (the *seed*),
2. **personalized PageRank** over the graph, personalized toward the query-similar seeds, so
   relevance diffuses across edges (multi-hop),
3. blend `final = α·cosine + (1−α)·pagerank` (α=0.5), return top-k.

**Visualization** (`src/graphrag/visualize.py`) — aggregates the chunk graph to a **page graph**
(node = page, size ∝ chunks, edge = cross-page relation) and renders:
- static **PNG** (Matplotlib): `docs/viz/page_graph.png` (committed)
- interactive **HTML** (Plotly): `docs/viz/page_graph.html` (git-ignored — ~4.7 MB inline JS;
  regenerate with `python -m src.graphrag.viz_cli --pdf <pdf>`)

## How to run

```bash
# select the graph retriever (default is hybrid)
RETRIEVER=graph python main.py --pdf data/earnings_presentation_q2fy26.pdf

# regenerate the visualization
python -m src.graphrag.viz_cli --pdf data/earnings_presentation_q2fy26.pdf   # (helper below)

# unit tests (no LLM)
python -m pytest tests/test_graphrag.py -q
```

Selection is via `RETRIEVER=hybrid|graph` (`src/config.py` → `build_retriever` factory). The
LangGraph agent, governance, and eval harness are unchanged — GraphRAG is just another `Retriever`.

## Results — honest comparison (hybrid vs graph)

Both retrievers run end-to-end through the same agent on the 7-question eval set (deterministic
metrics, no LLM judge, on Gemini answers):

| metric | hybrid (FAISS+BM25+rerank) | graph (PageRank) |
|--------|:-------------------------:|:----------------:|
| behaviour_accuracy (answer/refuse correct) | **1.0** | **1.0** |
| citation_validity | **1.0** | **1.0** |
| facts_accuracy (gold numbers present) | **1.0** | **1.0** |
| retrieve latency (typical) | fast (FAISS ANN) | higher — embeds query + PageRank each call |

**It's a tie on quality — and graph costs more.** Both answered all five answer-questions with valid
citations and the right numbers, and both correctly refused the two negative controls. The graph
retriever surfaces *different* chunks (e.g. for total income both still put the correct `[p22:c35]`
first; for airport income the two pick different supporting chunks) but the end-to-end answer quality
is identical. On this single, small, dense document GraphRAG does **not** beat hybrid + cross-encoder
rerank, and it adds per-query cost. This is the expected result — see below for when it *would* win.

## When GraphRAG actually helps (and why it's marginal here)

- This is a **single 41-page deck → a small, dense graph** (63 chunks, ~500 semantic edges). When
  the semantic graph is near-complete, PageRank approaches uniform and can **promote well-connected
  but less-relevant** chunks over the single best one — hurting precision. Hybrid + cross-encoder
  rerank is hard to beat on a small corpus.
- GraphRAG's payoff shows up on **large, multi-document corpora** and **entity-centric / multi-hop**
  questions ("how does X in doc A relate to Y in doc B?"), where traversing explicit entity/reference
  edges surfaces context that flat vector search misses (cf. Microsoft GraphRAG's community
  summaries). It also enables **global/thematic** questions via community detection.
- Practical next step for real value: build the graph over **entities/relations** (LLM-extracted
  triples) across many documents, not just chunk-similarity within one — and add community
  summaries. That's a larger effort tracked as a future extension.
