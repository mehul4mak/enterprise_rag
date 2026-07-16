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

## Phase 4B — Cost & Latency Optimization (this step)

### The idea in one line
Now that we can *measure* quality (4A), make the system **cheaper and faster** — and use the eval
harness to prove we didn't break anything doing it.

### Concept 1 — you can't optimize what you can't see (cost instrumentation)
We already time each pipeline stage. We added a **cost** number next to the time: for every LLM call
we estimate input/output **tokens** and multiply by a **price table** to get `cost_usd`. Now each
`/chat` response and the `/metrics` dashboard show what a query actually costs (~**$0.00013** for one
grounded answer here).

> Lesson: cost is just another metric. Put it on the same trace/span as latency and optimization
> becomes a data-driven decision, not a guess.

### Concept 2 — the semantic cache (the big win)
If someone asks a question we've already answered for this document, don't pay to answer it again.
We embed the query and, if it's within a cosine threshold of a past query, return the **cached
answer** — skipping retrieval and the LLM entirely.

"**Semantic**" is the key word: it's not exact-string matching. A paraphrase
("...income **for** H1-26" vs "...**in** H1-26", cosine 0.9935) still hits.

**Measured:** a cross-session repeat went from **8.05 s → 0.13 s (≈60×) and $0.00013 → $0**.

Two correctness rules that matter:
- **Per-document** cache (answers never leak between documents).
- Only **standalone** (no chat-history) questions are cached — a follow-up like "break that down"
  depends on history, so it must not be served from a query-only cache. The payoff is therefore
  **cross-session** (many users asking the same FAQ hit one shared cache).

> Lesson: caching an LLM system is mostly a *correctness* problem, not a storage one. Decide exactly
> when a cached answer is *safe* to reuse (same doc, no conversational context) before you cache.

### Honest calibration finding
My first test reused **one** chat session, so every question after the first had history → the cache
(correctly) refused to serve them, and it looked broken. It wasn't — I was testing the wrong scenario.
Re-testing with two sessions showed the 60× win. Also, the 0.97 threshold is deliberately strict:
close paraphrases hit, looser rewordings correctly miss and regenerate.

### How to run
```bash
SEMANTIC_CACHE=on python main.py --pdf data/earnings_presentation_q2fy26.pdf
# ask the same question in two runs → second is instant + free
```

### Files
- `src/observability/cost.py` — token/price estimation.
- `src/optimize/semantic_cache.py` — the per-document semantic cache.
- graph wiring in `src/graph.py` (cost on spans; cache in `RAGGraphAgent.ask`).
- Full write-up + numbers: [`../OPTIMIZATION.md`](../OPTIMIZATION.md).

### What's next
Model tiering (cheap model for query-rewriting), prompt compression, adaptive top-k — each guarded by
the eval harness. Then Phase 4C (PII dual-store vault) and 4A-adapters (multi-LoRA fine-tuning).

---

## Phase 7 — Retrieval Experiments (measuring what I'd only reasoned about)

### Why this phase exists
Two fair challenges: *"how do you know that chunking works well — did you try others?"* and *"are you
doing contextual retrieval or HyDE?"* Honest answers: I picked chunking by **reasoning, not
measurement**, and I'd done **neither** technique. So I measured.

### The trick that made it cheap
Scoring an *answer* needs an LLM. Scoring **retrieval** doesn't. Build a **gold set** (question → the
page that really contains the answer, verified by finding the answer string in the PDF), then measure:
- **hit@5** — did a gold page make the top-5? (what matters: we hand the LLM 5 chunks)
- **MRR** — 1/rank of the first gold page (ranking precision)

That's ~15 config sweeps for **zero LLM cost**.

> Lesson: **separate what you can measure for free from what costs money.** Most RAG tuning is a
> retrieval problem, and retrieval is free to evaluate.

### The three findings that surprised me
1. **BM25 (keyword) is the workhorse, not embeddings.** Alone it has the *best ranking* (MRR 0.833) —
   better than the full hybrid+rerank stack (0.708). Dense embeddings alone are weak (MRR 0.347).
   Makes sense in hindsight: a financial deck is discriminated by exact tokens (`H1-26`, `EBITDA`,
   `44,281`) — BM25's home turf, embeddings' weakness. The reranker still earns its keep: it's the
   only config that never misses (**hit@5 = 1.000**).
2. **My shipped chunking wasn't optimal.** `chunk_size=1000` was validated (1000–1500 = hit@5 1.000;
   2000 degrades), but my `overlap=150` scored the *worst* in its own row (0.708 vs 0.806 @300).
   The part I reasoned about held; the part I guessed didn't.
3. **Contextual retrieval beat HyDE — and it's free.** Prepending `[Context: <doc>, page N]` to each
   chunk lifted MRR 0.708 → **0.792** with no LLM. HyDE (LLM writes a fake answer, embed *that*) only
   reached 0.750 *and* costs a call per query — because it improves the **dense** probe, which is the
   weakest signal here.

> Lesson: **advanced techniques are corpus-dependent.** HyDE is a real technique that helps when
> embeddings carry the load (prose, contracts). On a keyword-dense deck it barely moves. "Is it
> state-of-the-art?" is the wrong question; "does it help *this* corpus?" is the right one.

### The honesty caveat
**n=6 questions.** One question = 0.167 of hit@5; an 0.08 MRR delta is half a rank on one question.
The robust signals (hit@5 plateau, BM25 > dense) are real; the fine-grained MRR orderings are
**within noise**. A production decision needs 50–100+ gold questions.

Full numbers + reproduce commands: [`RETRIEVAL_EXPERIMENTS.md`](RETRIEVAL_EXPERIMENTS.md).

### Phase 7 addendum — the result I almost mis-reported
All three embedding models scored **identically** (1.000 / 0.708) with the full stack. Identical
numbers are a **bug smell**, so instead of reporting "embedding choice doesn't matter" I isolated
**dense-only** — and they *do* differ a lot (bge-small 1.000/0.500 vs MiniLM 0.833/0.347). So it
wasn't a bug: **the cross-encoder reranker washes out the embedding choice**, because with only 63
chunks half the corpus enters the fusion pool anyway and the reranker decides the final order.

Two lessons:
- **When a result looks too clean, try to disprove it before you publish it.** A one-line diagnostic
  turned "suspicious identical numbers" into the most interesting finding of the phase.
- **The finding is scale-dependent and I said so:** on 63 chunks the reranker dominates; on 63,000 the
  dense recall would decide everything and bge-small's edge would be the whole ballgame.
