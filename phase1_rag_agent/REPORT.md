# Phase 1 — Conversational RAG Agent: Steps, Reasoning & Results

**Author:** (candidate submission for Adani AI Labs — Engineering Manager, Gen AI)
**Date:** 2026-07-13
**Deliverable:** A runnable, document-grounded, multi-turn RAG agent that answers questions
about a PDF with citations and clean refusals.

---

## 1. Understanding of the task (what was actually asked)

The assessment brief asks for a **working RAG chatbot over a PDF** with four hard requirements
and a set of "must-pass" acceptance tests. Reading between the lines, the brief is really testing
**four engineering judgments**, not just "can you call an LLM":

1. **Grounding discipline** — the agent must *refuse* when the answer isn't in the document
   ("Not found in the document."). The negative-control test (CEO's email) exists specifically to
   catch systems that hallucinate to please the user.
2. **Traceability** — every claim must cite `[p<page>]` or `[p<page>:c<chunk>]`. This is an
   enterprise/compliance signal (the JD stresses "assured lineage" and "sovereign" data).
3. **Retrieval quality** — numeric and cross-section questions only pass if the *right* chunk is
   retrieved. This tests chunking + retrieval, not model size.
4. **Generality** — "must work for similar PDFs (reports/decks/policies/contracts)" and
   "no hardcoding answers specific to the sample document."

So the design goal I set: **make retrieval do the heavy lifting and constrain the LLM to
extract-and-cite only.** That makes the system honest, portable, and cheap.

---

## 2. How I approached it (and why each choice)

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| PDF parsing | **PyMuPDF** (`fitz`) per-page text | Best free layout/text quality; gives reliable page numbers for citations |
| Chunking | **Per-page**, char-budgeted, line-packed with overlap | Guarantees every chunk maps to exactly one page → clean citations. Keeping a slide's text together keeps labels ("Total Income") with their numbers |
| Embeddings | **`all-MiniLM-L6-v2`** (local) | Tiny, fast on CPU, no API/egress → offline + sovereign-friendly |
| Vector index | **FAISS** (`IndexFlatIP`, cosine) | Exact search; dataset is small (tens–hundreds of chunks) so no ANN needed |
| Sparse index | **BM25** (`rank_bm25`) | Catches exact tokens (fiscal codes, "H1-26", "EBITDA") that embeddings blur |
| Fusion | **Reciprocal Rank Fusion** | Rank-based, scale-free — no need to normalize incompatible BM25/cosine scores |
| Reranking | **cross-encoder `ms-marco-MiniLM-L-6-v2`** | Query-aware re-scoring; measurably fixed period/section confusion (see §4) |
| LLM | **Local Ollama (default)**, pluggable API | Runs with no key/spend; swappable to Gemini for quality + Phase-2 alignment |
| Grounding | Strict system prompt + **post-hoc citation validation** | Belt-and-suspenders: if the model emits no valid citation, we force the refusal |
| Follow-ups | **Query condensation** using chat history | Turns "break that down…" into a standalone retrievable query |

Full rationale for the model/compute spend is in
[COST_AND_MODEL_RESEARCH.md](COST_AND_MODEL_RESEARCH.md).

---

## 3. Pipeline walkthrough

1. **Ingest** (`src/ingest.py`): extract text per page → pack lines into ~1000-char chunks with
   150-char overlap; each `Chunk` carries `page` + `chunk_id` → citation `[p22:c35]`.
2. **Index** (`src/index_store.py`): embed all chunks (FAISS) + tokenize (BM25). Persist to
   `.cache/<pdf-hash>` so re-runs are instant; the hash keys on file bytes + chunk params + model.
3. **Retrieve** (`src/retriever.py`): dense top-15 + sparse top-15 → RRF fuse → cross-encoder
   rerank → top-5.
4. **Condense** (`src/agent.py`): if there's history, rewrite the follow-up into a standalone query.
5. **Generate** (`src/prompts.py` + `src/llm.py`): feed the 5 cited context blocks under strict
   grounding rules; temperature 0.
6. **Validate** (`src/agent.py::_postprocess`): if the answer starts with "Not found", or has no
   citation, or cites nothing that was actually retrieved → return the canonical refusal.

---

## 4. Results

### 4.1 Retrieval quality (why hybrid + rerank matters)
For *"consolidated total income in H1-26"*, pure RRF ranked a Q2 chunk and an airport-cargo chunk
above the correct one. The cross-encoder reranker corrected this, placing the two right chunks
(`[p22:c35]` H1-26 Total Income, `[p2:c2]` consolidated highlights) at ranks 1–2 with a clear
score gap. This is the concrete payoff of the bonus retrieval features.

### 4.2 Acceptance tests

Run on the provided earnings deck with **`LLM_PROVIDER=gemini` / `gemini-flash-lite-latest`**
(full transcript: [`tests/acceptance_results_gemini.txt`](tests/acceptance_results_gemini.txt)):

| # | Scenario | Result | Evidence |
|---|----------|--------|----------|
| 1 | Grounded fact — major business segments | ✅ PASS | Lists Infra & Utility core, New Industries (Green H2), Primary Industry, Services (IRM/Mining), Direct-to-Consumer, Airports/Roads — each cited `[p11:c18]`, `[p2:c3]`, `[p26:c39]` |
| 2 | Numeric — consolidated total income H1-26 | ✅ PASS | "44,281 ₹ crore `[p22:c35]`" — correct value & period |
| 3 | Cross-section — EBITDA change drivers | ✅ PASS | IRM/Commercial Mining volume+price down; Airports/ANIL up — cited `[p2:c2]`, `[p22:c35]` |
| 4 | Negative control — CEO's email | ✅ PASS | "Not found in the document." (correct refusal) |
| 5 | Follow-up — airport → passenger/cargo | ✅ PASS | Condensed follow-up retrieved `[p16:c26]`: Pax 45.1→46.0 Mn (+4%), Cargo 5.5→5.7 L-MT (+4%) |

**5/5 passed.** The follow-up confirms chat-history-aware retrieval: "break that down into
passenger and cargo" carried no explicit subject, yet was rewritten into a standalone query that
retrieved the right page.

For reference, the same pipeline on the **local `gemma2:2b`** answered the numeric question
identically (`44,281 ₹ crore [p22:c35]`) but was slower (~90s/answer) and timed out on the broader
"list all segments" question — see [ISSUES_LOG.md](ISSUES_LOG.md) §6.4. This is the practical case
for the hosted model on the quality tier; the pipeline itself is provider-agnostic.

---

## 5. Honesty / limitations (see full audit in [ISSUES_LOG.md](ISSUES_LOG.md))

- **Slide tables aren't parsed structurally.** PyMuPDF's `find_tables()` returns 0 tables on the
  deck's slide layouts; numbers arrive as loose lines. Mitigated by keeping slide text together in
  one chunk, but a layout-aware parser (Phase 2) would be more robust.
- **Local 2B model is the quality floor, not ceiling.** The pipeline proves out on `gemma2:2b`;
  answer phrasing and multi-fact synthesis improve markedly with Gemini 2.5 Flash-Lite.
- **CPU latency.** ~30–60 s/answer locally (no GPU). Hosted API or a rented GPU removes this.
- **No boilerplate stripping** ("STRICTLY CONFIDENTIAL", menu chrome) — didn't hurt retrieval, so
  deferred.

---

## 6. Phase 2 — porting to GCP / Gemini (per the JD)

The JD centers on **GCP, Gemini Enterprise, the Gemini Agent Platform, Google ADK, LangGraph,
sovereign cloud, and data lineage.** Phase 1 was deliberately built so each component has a
clean GCP-managed counterpart:

| Phase 1 (local) | Phase 2 (GCP-native) | Notes |
|-----------------|----------------------|-------|
| PyMuPDF text extraction | **Document AI** (Layout/Form parser) | Structured tables, layout, OCR for scanned docs |
| `all-MiniLM` embeddings | **Vertex AI `text-embedding` (gemini-embedding)** | Managed, higher-dim, batched |
| FAISS | **Vertex AI Vector Search** (or AlloyDB/pgvector) | Managed ANN, scales to millions of chunks |
| BM25 (rank_bm25) | **Vertex AI Search** (built-in hybrid) or Elasticsearch on GKE | Managed hybrid + reranking |
| cross-encoder rerank | **Vertex AI Ranking API** | Managed reranker |
| Ollama `gemma2:2b` | **Gemini 2.5 (Vertex AI)** | Same `llm.py` seam; swap the client |
| hand-rolled agent loop | **Google ADK / LangGraph** graph | Nodes: condense → retrieve → ground → validate; matches JD frameworks |
| local `.cache/` | **GCS** for artifacts + **BigQuery** for eval logs | Lineage + governance |
| — | **Data lineage** via Dataplex + IAM access controls | Directly addresses JD's "assured lineage / governed datasets" |
| — | Deploy on **Cloud Run / GKE**, sovereign region | JD's scalability + sovereign-cloud KPIs (99.9% uptime, 50% more inference) |

The `src/llm.py` provider seam and the modular retriever mean this is a **migration, not a
rewrite** — each local component is swapped for its Vertex-managed equivalent behind the same
interface. A LangGraph/ADK rebuild of `agent.py` gives the observability, guardrails, and
multi-agent extensibility the enterprise role expects.

---

## Appendix — raw acceptance output

Full verbatim transcript (answers, per-question latency, and retrieved citations) is committed at
[`tests/acceptance_results_gemini.txt`](tests/acceptance_results_gemini.txt). Reproduce with:

```bash
LLM_PROVIDER=gemini python -m tests.acceptance --pdf data/earnings_presentation_q2fy26.pdf
```
