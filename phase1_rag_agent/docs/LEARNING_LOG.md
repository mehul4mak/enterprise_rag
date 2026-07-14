# Learning Log — a follow-along walkthrough

Plain-language summary of **what** we built each step, **why**, and the **key lesson** — so it can be
followed for learning. (The blow-by-blow engineering audit is in `ISSUES_LOG.md`; the deep design is
in `REPORT.md`. This file is the friendly tour.)

---

## The journey so far

| Phase | What | Why | Key lesson |
|------|------|-----|-----------|
| **1 — local RAG** | PDF → chunks → hybrid retrieval (FAISS + BM25) → rerank → grounded answer with citations + "Not found" refusal. FastAPI + Docker + CI. | A RAG chatbot is really an *honesty* problem: answer only from the doc, cite it, refuse otherwise. | **Retrieval does the heavy lifting; the LLM only extracts + cites.** A tiny model then works. |
| **2 — GCP-shaped, local** | Same behaviour behind **provider interfaces** (local↔GCP = a `BACKEND=` swap) and rebuilt as a **LangGraph** agent. Added governance: guardrails, Model Armor (PII), residency, lineage, tracing, metrics. | Make Phase 3 a *config swap*, not a rewrite; add the enterprise controls the JD asks for. | **Program to interfaces.** Once every cloud dependency is a seam, swapping infra is trivial. |
| **3 — real GCP** | Real Vertex/Document AI provider code + **Terraform** for `asia-south1` + Cloud Run/Cloud Build. Deploy-ready, not deployed (no creds/spend). | Prove the port is real without burning money. | **Lazy imports + clear config errors** let cloud code live in the repo and stay testable offline. |
| **4 — beyond RAG** | (this phase) Start with an **evaluation harness** — the quality gate everything else needs. | You can't optimize (cost/latency) or fine-tune safely without a way to *measure quality*. | **Measure before you optimize.** |

---

## Phase 4A — the Evaluation Harness (this step)

### The idea in one line
Before we make the system cheaper/faster or fine-tune it, we need an objective, repeatable way to
answer: **"is the RAG system actually giving good, grounded, correct answers?"**

### The concepts (plain English)

We score each answer on two kinds of check:

**1. Deterministic checks (free — no model needed):**
- **behaviour_correct** — did it *answer* when it should, and *refuse* when it should? (We include
  "negative control" questions whose answer isn't in the doc — a good system must say "Not found".)
- **citations_valid** — does every `[p22:c35]` in the answer actually point at a chunk we retrieved?
  (Catches invented citations.)
- **facts_present** — for numeric questions, is the gold number (e.g. `44,281`) literally in the answer?

**2. LLM-as-judge (one model call per answer):**
We ask a model to grade the answer *given the retrieved context*, returning JSON:
- **faithfulness** (0–1) — is every claim supported by the context, or did it fabricate?
- **answer_relevance** (0–1) — does it actually address the question?
- **citation_supported** (0/1) — do the cited passages really contain the claim?

> Why "LLM-as-judge"? Grading free-text answers by rule is brittle; a model reading the context can
> tell whether the answer is grounded. This is the same idea as **Vertex AI's Gen AI Evaluation
> Service**. We combine the three scores into **one** call to keep cost/tokens low.

### Gates (turning scores into a pass/fail)
Aggregate metrics must clear thresholds (`src/eval/harness.py:GATES`):
- behaviour accuracy = 1.0 (no wrong answer/refuse decisions)
- citation validity = 1.0 (no invented citations)
- faithfulness mean ≥ 0.80, answer-relevance mean ≥ 0.80

If all gates pass, the build is "good enough" — this is what a CI regression gate or a fine-tuning
"is the new model better?" check would use.

### How to run it
```bash
# deterministic + LLM-judge (needs an LLM; sparing on the free tier)
python -m src.eval.harness --pdf data/earnings_presentation_q2fy26.pdf

# deterministic only (no judge calls)
python -m src.eval.harness --no-judge
```
It writes `eval_report.json` + `eval_report.md`.

### Files
- `src/eval/dataset.jsonl` — the golden questions (with expected answer/refuse + required facts).
- `src/eval/judges.py` — deterministic checks + the combined LLM-judge.
- `src/eval/harness.py` — runs the agent over the dataset, scores, gates, writes the report.

### Results

Run on the earnings deck with Gemini (`gemini-flash-lite-latest`), 7 questions — **all gates pass**:

| metric | value | gate |
|--------|------:|-----:|
| behaviour_accuracy (answer/refuse correct) | 1.0 | 1.0 ✅ |
| citation_validity | 1.0 | 1.0 ✅ |
| facts_accuracy (gold numbers present) | 1.0 | — |
| faithfulness_mean (judge) | 1.0 | 0.80 ✅ |
| answer_relevance_mean (judge) | 1.0 | 0.80 ✅ |

Both negative controls (`ceo_email`, and a new `share_price` market-data question) correctly
**refused**; every numeric gold (`44,281`, `46.0`, `5.7`, `5,882`) was present. Full report:
[`eval_report.md`](../eval_report.md).

> **Honest caveat — self-judging bias.** The judge here is Gemini grading Gemini's *own* answers
> (same model family), which tends to be lenient — so the perfect faithfulness/relevance 1.0s are
> softer evidence than they look. The **stronger** signal is the *deterministic* metrics
> (behaviour, citation validity, gold-fact presence), which are objective and independent of the
> judge — and those also passed. Production hardening: use a **different/stronger judge model**, add
> **human spot-checks**, and add **adversarial** questions designed to induce hallucination. This is
> a general LLM-as-judge lesson: never let the model being tested also be its own grader without a
> second, independent check.

### What this unlocks (next Phase 4 steps)
- **B — cost/latency optimization**: safe to cut cost (model tiering, caching) because this harness
  tells us if quality dropped.
- **A — fine-tuning / multi-LoRA**: the harness is the "did the new adapter actually help?" judge,
  and the per-question records become training/preference data.
- **C — PII vault**: eval also guards against a redaction change breaking answers.

---

## Phase 5 — GraphRAG (page graph + graph retrieval)

### The idea in one line
Instead of treating the document as a flat list of chunks, build a **graph** (chunks connected by
adjacency, semantic similarity, and shared keywords) and retrieve by **walking** it — plus draw the
pages as a graph you can look at.

### How it works (plain English)
1. **Build the graph** — every chunk is a node; we connect two chunks if they're on the same page,
   look similar (embedding cosine), or share keywords. Weighted edges.
2. **Retrieve** — embed the question, find the most-similar "seed" chunks, then run **personalized
   PageRank** so relevance *flows* to neighbours (multi-hop). Blend PageRank with plain similarity.
3. **Visualize** — collapse the chunk graph to a **page graph** (node = page) and render a PNG +
   interactive HTML.

### Why we tried it, and the honest result
Graph retrieval shines when answers require **connecting facts across many documents** or following
**entity/reference links** (Microsoft's GraphRAG, multi-hop QA). We tested whether it helps *here*.

**Result — a tie, and graph costs more.** On the 7-question eval set, hybrid and graph *both* scored
behaviour 1.0 / citation-validity 1.0 / facts 1.0. They retrieve different chunks, but the answers
come out equally correct — and the graph retriever is slower (it embeds the query and runs PageRank
every call). So on this one small deck, GraphRAG buys nothing over hybrid + rerank. A **well-measured
negative result is still a result** — we now know *not* to reach for GraphRAG on small single docs.

### Key lesson
**Match the retrieval method to the corpus.** On one small, dense document, a graph adds little (and
PageRank can even promote well-connected-but-off-topic chunks over the single best one). GraphRAG
earns its keep on **large, multi-document, entity-centric** knowledge bases — build the graph over
extracted **entities/relations**, not just chunk-similarity within a single file.

### Files
- `src/graphrag/graph_build.py` — build the weighted chunk graph + page graph.
- `src/graphrag/retriever.py` — `GraphRetriever` (personalized PageRank), selected by `RETRIEVER=graph`.
- `src/graphrag/visualize.py` + `viz_cli.py` — page-graph PNG/HTML.
- `docs/GRAPHRAG.md` — design + results; `docs/viz/page_graph.png` — the picture.
