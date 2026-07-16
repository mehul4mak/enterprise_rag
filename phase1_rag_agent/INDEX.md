# Enterprise RAG — Documentation Index

Start here. This maps every document, branch, and module so you can navigate the project by intent.

> **What it is:** a document-grounded conversational RAG agent (answer only from the PDF, with page
> citations, refusing when unsupported) — grown across six phases from a local prototype to a
> deploy-ready, governed, evaluated, GCP-portable platform. Built for the Adani Gen-AI EM assessment.

---

## Read in this order

| # | Document | What you'll get |
|---|----------|-----------------|
| 1 | [README.md](README.md) | Quickstart — run it in 3 commands; what maps to the task spec |
| 2 | [docs/LEARNING_LOG.md](docs/LEARNING_LOG.md) | **Plain-language tour** of every phase (what / why / key lesson) |
| 3 | [docs/QA_EXPLAINED.md](docs/QA_EXPLAINED.md) | **Q&A** — condensing, prompts.py, models, the two total-income pages, when it refuses, HyDE/contextual, chunking evidence |
| 4 | [REPORT.md](REPORT.md) | The engineering design + reasoning + acceptance results + GCP migration map |
| 5 | [docs/SOCRATIC_REVIEW.md](docs/SOCRATIC_REVIEW.md) | Interviewer-grade self-review: findings, honest limits, JD alignment |
| 6 | [ISSUES_LOG.md](ISSUES_LOG.md) | The full audit trail — **every** bug/decision, including the failures |

## Deep-dive docs (by topic)

| Topic | Document |
|-------|----------|
| Hosting / deploy / API reference | [DEPLOYMENT.md](DEPLOYMENT.md) |
| Governance (guardrails, Model Armor, residency, lineage, tracing) → GCP mapping | [GOVERNANCE.md](GOVERNANCE.md) |
| Model & compute cost analysis (GPU vs API) | [COST_AND_MODEL_RESEARCH.md](COST_AND_MODEL_RESEARCH.md) |
| Visual-PDF parsing: 5 methods compared | [PARSING.md](PARSING.md) |
| GCP deploy runbook (Terraform + Cloud Run, asia-south1) | [GCP_SETUP.md](GCP_SETUP.md) |
| Latency & cost optimization (cost tracking + semantic cache) | [OPTIMIZATION.md](OPTIMIZATION.md) |
| GraphRAG design + honest hybrid-vs-graph comparison | [docs/GRAPHRAG.md](docs/GRAPHRAG.md) |
| Retrieval experiments: chunking/embedding sweeps, HyDE, contextual retrieval | [docs/RETRIEVAL_EXPERIMENTS.md](docs/RETRIEVAL_EXPERIMENTS.md) |
| Phase 4 roadmap (multi-LoRA, PII vault, non-RAG agents) | [PHASE4_PLAN.md](PHASE4_PLAN.md) |

---

## Phases → branches

Each phase is its own branch; **`phase-6-final`** consolidates everything.

| Phase | Branch | Adds | One-line lesson |
|-------|--------|------|-----------------|
| 1 | `phase-1-local-rag` | Local RAG (hybrid retrieval + rerank + grounding) + FastAPI + Docker + CI | Retrieval does the heavy lifting; LLM only extracts + cites |
| 2 | `phase-2-gcp-local` | Provider interfaces + LangGraph agent + governance layer | Program to interfaces → infra swap, not rewrite |
| 3 | `phase-3-gcp` | Real Vertex/Document AI code + Terraform (asia-south1) + Cloud Run | Lazy imports + clear errors keep cloud code testable offline |
| 4A | `phase-4-eval` | Eval harness (LLM-judge + deterministic gates) | Measure before you optimize |
| 4B | `phase-4b-optimization` | Cost instrumentation + semantic cache (60× / $0) | Caching an LLM app is a correctness problem |
| 5 | `phase-5-graphrag` | Page-graph + PageRank retriever + visualization | Fancier ≠ better; graph wins on large multi-doc corpora |
| — | **`phase-6-final`** | **Merge of all above + this index + Socratic review** | — |

---

## Code map (`src/`)

```
ingest.py          PDF → page-tagged chunks
embeddings.py      local sentence-transformers
index_store.py     FAISS + BM25, disk-cached (696× warm reload)
retriever.py       RRF fusion + cross-encoder rerank + structured page lookup
grounding.py       citation validation / refusal  (shared, single source of truth)
prompts.py         grounding + follow-up condensation prompts
llm.py             ollama / openai / anthropic / gemini backends
graph.py           LangGraph agent: guard→condense→retrieve→generate→validate→guard→finalize
pipeline.py        assemble agent from a PDF via the provider factory
agent.py           Phase-1 hand-rolled agent (kept for reference/tests)
api.py             FastAPI service (/chat /ingest /health /metrics /sessions)

providers/         the local↔GCP seam
  base.py            interfaces: DocumentParser, Embedder, Retriever, LLMProvider
  local.py           local implementations
  gcp.py             Vertex / Document AI implementations (deploy-ready)
  parsers.py         pymupdf / pdfplumber / docling / easyocr / VLM
  factory.py         BACKEND=local|gcp, PARSER=…, RETRIEVER=hybrid|graph

governance/        guardrails.py · model_armor.py · residency.py · lineage.py
observability/     tracing.py · metrics.py · cost.py · logging_setup.py
optimize/          semantic_cache.py
eval/              dataset.jsonl · judges.py · harness.py
graphrag/          graph_build.py · retriever.py · visualize.py
```

## Run it

```bash
pip install -r requirements.txt                       # + requirements-parsers.txt / -gcp.txt as needed
python main.py --pdf ./data/earnings_presentation_q2fy26.pdf     # CLI chat
uvicorn src.api:app --port 8000                       # HTTP service (Swagger at /docs)
python -m tests.acceptance --pdf data/earnings_presentation_q2fy26.pdf   # 5 mandated tests
python -m src.eval.harness                            # gated quality report
pytest tests/ -q --ignore=tests/acceptance.py --ignore=tests/parser_benchmark.py   # 49 offline tests
```

Switches (all env): `BACKEND=local|gcp` · `LLM_PROVIDER=ollama|gemini|openai|anthropic` ·
`PARSER=pymupdf|pdfplumber|docling|easyocr|vlm` · `RETRIEVER=hybrid|graph` ·
`SEMANTIC_CACHE=on` · `GUARDRAILS=on` · `DATA_RESIDENCY=off|regional|strict`.
