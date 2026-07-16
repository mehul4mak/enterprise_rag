# Enterprise RAG — Document-Grounded Conversational Agent

A runnable, document-grounded conversational agent. It ingests a PDF, builds a **hybrid
(dense + sparse) retrieval index with cross-encoder reranking**, and answers multi-turn
questions **only from retrieved context**, with **page + chunk citations** and a strict
**"Not found in the document."** refusal when the answer isn't supported.

Built for the Adani Gen-AI assessment. Runs **fully offline by default** (local Ollama), and
switches to Gemini / OpenAI / Anthropic with a one-line `.env` change. Grown across six phases from a
local prototype to a governed, evaluated, GCP-portable platform.

> 🧭 **New here? Start with [INDEX.md](INDEX.md)** — it maps every doc, branch, and module.
> For the plain-language tour see [docs/LEARNING_LOG.md](docs/LEARNING_LOG.md); for the
> interviewer-grade self-review see [docs/SOCRATIC_REVIEW.md](docs/SOCRATIC_REVIEW.md).

> **Branches:** `phase-1-local-rag` = local RAG agent + hosting/CI. `phase-2-gcp-local` = same
> behaviour re-architected as a **LangGraph** agent behind **provider interfaces** + governance
> (guardrails/Model Armor/residency/lineage/tracing). `phase-3-gcp` = **deploy-ready** GCP backend
> (Document AI + Vertex Vector Search/Ranking + Gemini-on-Vertex) with Terraform IaC for
> `asia-south1` — see [GCP_SETUP.md](GCP_SETUP.md). Switch backends with `BACKEND=local|gcp`.
> Future: [PHASE4_PLAN.md](PHASE4_PLAN.md) (multi-LoRA, cost/latency optimization, PII vault, non-RAG GenAI).

---

## Quickstart

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. (default) Local model — no API key needed. Requires Ollama running:
ollama serve &          # if not already running
ollama pull gemma2:2b   # the default local model

# 3. Run the chat loop against a PDF
python main.py --pdf ./data/earnings_presentation_q2fy26.pdf
```

Inside the chat loop:
- ask any question about the document
- `:debug` — toggle the retrieval-visibility panel (top-k snippets + scores + citations)
- `:quit` — exit

### Use a hosted model instead (recommended for best quality)

```bash
cp .env.example .env
# edit .env:
#   LLM_PROVIDER=gemini
#   GOOGLE_API_KEY=<your free key from https://aistudio.google.com/apikey>
python main.py --pdf ./data/earnings_presentation_q2fy26.pdf
```

See **[COST_AND_MODEL_RESEARCH.md](COST_AND_MODEL_RESEARCH.md)** for the model/compute
cost analysis and recommendation.

---

## What it does (maps to the task spec)

| Requirement | Where |
|-------------|-------|
| A) Ingestion + indexing (per-page text, chunk + page metadata, retrieval index) | `src/ingest.py`, `src/index_store.py` |
| B) Multi-turn conversational Q&A | `src/agent.py`, `main.py` |
| C) Grounded answers with citations `[p13]` / `[p13:c42]` + "Not found" refusal | `src/agent.py`, `src/prompts.py` |
| D) Retrieval visibility (top-k snippets + scores) | `main.py :debug`, `src/retriever.py` |
| Bonus: Hybrid retrieval (BM25 + vectors) + reranking + **structured page lookup** | `src/retriever.py` |

**Retrieval is three-way:** semantic (FAISS), keyword (BM25), and **structured/metadata** — a
positional query like *"what is on page 3"* is served by looking up the `page` field directly
(neither text retriever can, since the page number isn't a word in the chunk).
| Phase 2: governance (guardrails, Model Armor, residency, lineage, tracing, metrics) | `src/governance/`, `src/observability/`, [GOVERNANCE.md](GOVERNANCE.md) |
| Phase 2: multi-method visual parsing (pymupdf/pdfplumber/docling/OCR/VLM) | `src/providers/parsers.py`, [PARSING.md](PARSING.md) |

---

## Architecture

```
PDF ──▶ ingest.py ──▶ chunks (page-tagged)
                          │
                          ├──▶ embeddings (MiniLM) ──▶ FAISS  ─┐
                          └──▶ BM25 (rank_bm25)             ─┤
                                                             ▼
        question ──▶ [condense follow-up] ──▶ retriever: RRF fuse ──▶ cross-encoder rerank ──▶ top-k
                                                             │
                                                             ▼
                       grounded prompt (context + citations) ──▶ LLM ──▶ answer
                                                             │
                                                             ▼
                        citation validation / refusal (agent._postprocess)
```

Key design choice: **retrieval does the heavy lifting; the LLM only extracts and cites.**
That's why even a small local model answers numeric questions correctly and refuses cleanly.

---

## Run as a hosted service (API)

```bash
export LLM_PROVIDER=gemini GOOGLE_API_KEY=...   # or LLM_PROVIDER=ollama
export INDEX_PDF=./data/earnings_presentation_q2fy26.pdf
uvicorn src.api:app --host 0.0.0.0 --port 8000  # Swagger UI at /docs
```
Or containerized: `docker compose up --build`. Full details in **[DEPLOYMENT.md](DEPLOYMENT.md)**.

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"question":"What is the consolidated total income in H1-26?"}' | jq
# → {"answer":"... 44,281 ₹ crore [p22:c35].","citations":["[p22:c35]"], ...}
```

## Run the tests

```bash
python -m pytest tests/test_chunking.py tests/test_grounding.py tests/test_api.py -q   # offline unit tests
python -m tests.acceptance --pdf data/earnings_presentation_q2fy26.pdf                 # 5 mandated scenarios (needs LLM)
```

Results of the 5 mandated scenarios (5/5 pass on Gemini) are committed at
[`tests/acceptance_results_gemini.txt`](tests/acceptance_results_gemini.txt) and summarized in
[REPORT.md](REPORT.md) §4.2.

## CI/CD

- `.github/workflows/ci.yml` — lint (ruff) + offline unit tests on every push/PR.
- `.github/workflows/docker.yml` — builds the container (pushes to GHCR on version tags).

---

## Project layout

```
phase1_rag_agent/
├── main.py                     # single-command CLI entrypoint (chat loop)
├── requirements.txt            # runtime deps
├── requirements-dev.txt        # + pytest, ruff, httpx
├── pyproject.toml              # ruff + pytest config
├── Dockerfile                  # container image (pre-caches models, non-root)
├── docker-compose.yml          # host the API (+ optional local ollama)
├── .dockerignore
├── .env.example                # provider config (copy to .env)
├── README.md
├── REPORT.md                   # steps, reasoning, results, Phase-2 plan
├── DEPLOYMENT.md               # run/host/CI guide + API reference
├── COST_AND_MODEL_RESEARCH.md  # model & GPU cost analysis
├── ISSUES_LOG.md               # full engineering audit log
├── data/                       # PDFs (earnings deck + 2 samples)
├── src/
│   ├── config.py               # all tunables + provider selection
│   ├── ingest.py               # PDF → page-tagged chunks
│   ├── embeddings.py           # local sentence-transformers
│   ├── index_store.py          # FAISS + BM25 build/persist/reload
│   ├── retriever.py            # RRF fusion + cross-encoder rerank
│   ├── llm.py                  # ollama/openai/anthropic/gemini backends
│   ├── prompts.py              # grounding + condensation prompts
│   ├── agent.py                # Phase 1 multi-turn orchestration (hand-rolled)
│   ├── grounding.py            # shared citation-validation / refusal logic
│   ├── graph.py                # Phase 2 LangGraph agent (condense→retrieve→generate→validate)
│   ├── pipeline.py             # assemble agent from a PDF via the provider factory
│   ├── providers/              # Phase 2 provider abstraction (local ↔ GCP swap)
│   │   ├── base.py             #   interfaces: DocumentParser, Embedder, Retriever, LLMProvider
│   │   ├── local.py            #   local impls (wrap Phase 1 components)
│   │   ├── gcp.py              #   Vertex / Document AI stubs (Phase 3 target)
│   │   └── factory.py          #   selects impls from BACKEND=local|gcp
│   └── api.py                  # FastAPI service (hosting)
└── tests/
    ├── test_chunking.py        # chunking unit tests
    ├── test_grounding.py       # refusal / citation-validation unit tests
    ├── test_api.py             # FastAPI smoke tests
    ├── acceptance.py           # 5 mandated scenarios (needs LLM)
    └── acceptance_results_gemini.txt   # committed run transcript (5/5 pass)
```

CI/CD workflows live at the **repo root** in `.github/workflows/`.
