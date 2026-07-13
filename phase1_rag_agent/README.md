# Conversational RAG Agent over PDFs (Phase 1)

A runnable, document-grounded conversational agent. It ingests a PDF, builds a **hybrid
(dense + sparse) retrieval index with cross-encoder reranking**, and answers multi-turn
questions **only from retrieved context**, with **page + chunk citations** and a strict
**"Not found in the document."** refusal when the answer isn't supported.

Built for the Adani Gen-AI assessment. Runs **fully offline by default** (local Ollama), and
switches to Gemini / OpenAI / Anthropic with a one-line `.env` change.

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
| Bonus: Hybrid retrieval (BM25 + vectors) + reranking | `src/retriever.py` |

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

## Run the tests

```bash
python -m pytest tests/test_chunking.py -q      # unit tests
python -m tests.acceptance --pdf data/earnings_presentation_q2fy26.pdf   # 5 mandated scenarios
```

---

## Project layout

```
phase1_rag_agent/
├── main.py                     # single-command entrypoint (chat loop)
├── requirements.txt
├── .env.example                # provider config (copy to .env)
├── README.md
├── REPORT.md                   # steps, reasoning, results, Phase-2 plan
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
│   └── agent.py                # multi-turn orchestration + refusal logic
└── tests/
    ├── test_chunking.py
    └── acceptance.py
```
