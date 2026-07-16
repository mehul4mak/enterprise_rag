# Q&A — how the pipeline actually works

Direct answers to the questions asked while reviewing the code, each traced to the file that
implements it. Companion to [LEARNING_LOG.md](LEARNING_LOG.md) (the narrative tour) and
[RETRIEVAL_EXPERIMENTS.md](RETRIEVAL_EXPERIMENTS.md) (the measurements behind Q5/Q6).

**Questions answered here**

1. [What is condensing, and why is it needed?](#1-what-is-condensing-and-why-is-it-needed)
2. [What is `prompts.py` for?](#2-what-is-promptspy-for)
3. [Which models are used, and where?](#3-which-models-are-used-and-where)
4. [The deck has two H1-26 total-income figures — which is correct?](#4-the-deck-has-two-h1-26-total-income-pages--which-is-correct)
5. [When does it say "Not found in the document"?](#5-when-does-it-say-not-found-in-the-document)
6. [Are we doing Contextual Retrieval or HyDE?](#6-are-we-doing-contextual-retrieval-or-hyde)
7. [How do you know that chunking works? Why not try several?](#7-how-do-you-know-that-chunking-works-why-not-try-several)

---

## 1. What is condensing, and why is it needed?

**File:** `src/agent.py` → `RAGAgent._condense()` · prompt in `src/prompts.py` → `SYSTEM_CONDENSE`

Retrieval is **stateless**. It sees one query string and nothing else. That breaks the moment a
conversation becomes natural:

```
Turn 1  User: "What was the total income in H1-26?"        ← retrievable
Turn 2  User: "And break that down by segment."            ← retrieves NOTHING useful
```

Turn 2 has no content to match on. "break that down by segment" contains no `H1-26`, no `total
income` — BM25 has no keywords to score and the embedding of that sentence points nowhere near the
right page. The pronoun ("that") carries the meaning, and the pronoun's referent lives in the
**history**, not the query.

Condensing fixes this by asking the LLM to rewrite the follow-up into a **standalone** query using
recent turns, *before* retrieval runs:

> `"And break that down by segment."` → `"Adani Enterprises total income H1-26 broken down by business segment"`

That rewritten string is what gets retrieved with. The **original** question is still what the
answering model sees — we rewrite the *search*, not the *ask*.

**Three deliberate design choices**, all visible in `_condense()`:

| Choice | Why |
|---|---|
| **Skipped entirely when `history` is empty** | The first question is already standalone. No history → no LLM call → zero added latency/cost on single-shot use. |
| **Any exception falls back to the raw question** | Condensing is an *optimisation*, not a dependency. A rate limit or a network blip must never break the turn. |
| **Rewrite rejected if empty or >300 chars** | Guards against a chatty model that ignores "output ONLY the query" and starts answering instead. A 2-paragraph "rewrite" is a failure, so we discard it. |

Only the last `max_history_turns` turns are passed in — history is truncated so a long conversation
can't grow the prompt without bound.

> **Side-effect worth knowing:** because condensing calls the LLM, the semantic answer cache
> (Phase 4B) is disabled once history exists. That's intentional — caching follow-ups whose meaning
> depends on the conversation would return wrong answers.

## 2. What is `prompts.py` for?

**File:** `src/prompts.py` (42 lines, no imports, no logic)

It separates **policy** from **mechanism**.

- `agent.py` is the *mechanism*: condense → retrieve → generate → validate.
- `prompts.py` is the *policy*: what the model is allowed to do.

Everything that constrains model behaviour lives in that one file, so prompt engineering is a
single reviewable diff instead of strings scattered through the orchestration code:

| Constant / function | Role |
|---|---|
| `SYSTEM_GROUNDED` | The 5 grounding rules: context only, cite every factual sentence, exact refusal string, copy numbers verbatim, no preamble. |
| `build_qa_prompt()` | Assembles `CONTEXT` (citation-tagged passages) + `QUESTION`, and **repeats** the refusal instruction at the end. |
| `SYSTEM_CONDENSE` | "Rewrite into a standalone query. Output ONLY the query. Do not answer." |
| `build_condense_prompt()` | Formats history + follow-up. |

Two details in there are load-bearing rather than cosmetic:

- **Rule 4 — "copy numbers verbatim… do not compute"** — this is what stops the model from adding up
  two segment figures and presenting the sum as if the deck stated it. In a financial-QA system,
  a *plausible arithmetic result with a citation* is the most dangerous possible failure, because it
  looks exactly like a correct answer.
- **The refusal instruction appears twice** (system prompt rule 3 *and* the end of the user prompt).
  Small models drift from a system prompt over a long context; restating it adjacent to the question
  measurably improves compliance.

## 3. Which models are used, and where?

Four models, and **only one of them is remote**:

| Stage | Model | Where it runs |
|---|---|---|
| Embeddings (index + query) | `all-MiniLM-L6-v2` | **local** (sentence-transformers) |
| Reranking | `ms-marco-MiniLM-L-6-v2` cross-encoder | **local** |
| Sparse retrieval | BM25 | **local** (no model — pure statistics) |
| Generation + condensing | `gemini-flash-lite-latest` (`BACKEND=gcp`) or `gemma2:2b` via Ollama (repo default) | remote / local |

The important property: **only generation leaves the machine.** The document text is embedded,
indexed, searched and reranked locally. This is what makes `RESIDENCY=strict` (Phase 2) enforceable —
flip to the local LLM and *nothing* crosses a network boundary.

On the model IDs: `gemini-flash-lite-latest` is deliberately **version-agnostic**. Pinning
`gemini-2.5-flash-lite` broke with a 404 ("no longer available to new users") on a freshly created
key, and `gemini-2.0-flash` was deprecated on 2026-06-01 mid-project. The `-latest` alias survives
both. (Full incident: `ISSUES_LOG.md`.)

## 4. The deck has two H1-26 total-income pages — which is correct?

**Both. They agree.** This looked like a contradiction and isn't — it's the same total presented two
ways:

| Page | View | Arithmetic |
|---|---|---|
| **p2** | Breakdown **by business** | `27,109 + 17,172 = 44,281` |
| **p22** | **Y-o-Y waterfall** (last year → this year) | `49,263 − 4,867 − 115 = 44,281` |

Both reconcile to **₹44,281 crore**. p22's `49,263` is the *prior-year* figure being bridged down,
not a competing H1-26 number — that's what makes it look like a conflict when skimming.

So when the agent cites **both** `[p2]` and `[p22]`, that is the *correct* behaviour, not confusion.
This is also why the gold set (`src/experiments/goldset.py`) marks `total_income_h1` as
`gold_pages: [2, 22]` and counts a hit on **either** — scoring one of them "wrong" would be
penalising a right answer.

**The general lesson:** in financial decks the same fact recurs across summary, segment and bridge
slides. A retrieval system that assumes one-fact-one-page will mis-evaluate constantly.

## 5. When does it say "Not found in the document"?

**File:** `src/grounding.py` → `postprocess_answer()`

Three independent triggers. The first is the model's judgement; the other two are **checks applied
after generation**, which fire regardless of what the model claims:

1. **The model judges the context insufficient** and emits the refusal (`SYSTEM_GROUNDED` rule 3).
2. **The answer carries no citation at all** — `CITATION_RE` finds no `[p…]` tag. An uncited claim is
   an ungrounded claim, so it is discarded.
3. **The answer cites something that wasn't retrieved** — the tag doesn't match any chunk in
   `valid_citations(retrieved)`. This catches a *fabricated* citation, the failure mode where a model
   invents `[p7:c12]` because the format looks right.

Triggers 2 and 3 matter because they don't trust the model. The model can hallucinate an answer; it
cannot hallucinate its way past a set-membership check against the chunks actually retrieved.

**So why does it refuse a genuine document question?** Because **retrieval missed** — the answer was
never in the context, so refusal is the honest outcome. That's exactly what happened with
*"what is on page 1?"*:

> Page 1's tokens are `['adani','enterprises','limited','earnings','presentation','q2','fy26']` —
> containing **neither "page" nor "1"**. The page number is **metadata**, not text. BM25 had no token
> to match; FAISS had no meaning to match. Neither retriever *can* serve that query.

Fixed in `src/retriever.py` → `structured_lookup()`, a third retrieval path that queries the `page`
**field** and pins those chunks. `"what is on page 1?"` now answers correctly; `"revenue in 2024"`
doesn't false-trigger.

> **`"show me the index"` still refuses — and that is correct.** The deck has no table of contents.
> The right response to "show me something that doesn't exist" is to say so, not to synthesise one.
> Not every refusal is a bug; some are the system working.

## 6. Are we doing Contextual Retrieval or HyDE?

**Originally: neither, in any phase.** Fair hit. Both are now implemented and benchmarked in
Phase 7 (`src/experiments/advanced.py`).

**Contextual Retrieval** (Anthropic's technique) — prepend a blurb situating each chunk in the
document before indexing, so an isolated block of numbers stops being ambiguous:

```
[Context: Adani Enterprises Limited (AEL) Q2 FY26 earnings presentation, page 22.] <chunk text>
```

**HyDE** (Hypothetical Document Embeddings) — a *question* looks nothing like an *answer*, so
embedding it searches for the wrong shape of language. Instead ask the LLM to write a **fake
plausible answer** and embed *that*. Note it replaces **only the dense probe** — BM25 keeps the real
question, because its keywords are the signal.

| variant | hit@5 | MRR | cost |
|---|---:|---:|---|
| baseline (shipped) | 1.000 | 0.708 | — |
| **contextual-cheap** | 1.000 | **0.792** | **free** |
| hyde | 1.000 | 0.750 | +1 LLM call / query |

**Verdict: adopt contextual, skip HyDE here.** Contextual wins more *and* costs nothing. HyDE
under-delivers for a specific, explainable reason — it improves the **dense** probe, and §7 shows
dense is the *weakest* signal on this corpus. HyDE would shine on prose (contracts, policies) where
embeddings carry the load; not on a keyword-dense financial deck.

## 7. How do you know that chunking works? Why not try several?

**Also a fair hit.** Phase 1's `chunk_size=1000, overlap=150` was chosen by **reasoning** ("keep a
slide's label next to its numbers; one page per chunk for clean citations") — **not measurement**.
Phase 7 measured it.

**Method:** scoring an *answer* needs an LLM, but scoring *retrieval* doesn't. A gold set
(question → verified answer page) makes sweeps essentially free — `hit@5` and `MRR`, zero LLM calls.

**Result — my reasoning was half right:**

- ✅ **`chunk_size=1000` validated** — sits in the 1000–1500 sweet spot where **hit@5 = 1.000**. At
  2000 it degrades to 0.833 (chunks blur multiple topics).
- ❌ **`overlap=150` was the worst in its row** — 0.708 MRR vs **0.806** at overlap 300. The
  parameter I reasoned carefully about held; the one I guessed at didn't.

**And two findings I didn't expect:**

| Finding | Evidence |
|---|---|
| **BM25 alone ranks best** (MRR **0.833**) — better than the full hybrid+rerank stack (0.708). Dense alone is weak (0.347). | This is a *financial deck*: the discriminating tokens are exact strings (`H1-26`, `44,281`). BM25's home turf; general-purpose embeddings' weakness. The reranker still earns its place — it's the **only** config reaching hit@5 = 1.000. |
| **The embedding model is irrelevant while the reranker is on.** All three scored *identically*. | Identical numbers are a bug smell, so I checked rather than published: dense-only proves they genuinely differ (bge-small 1.000/0.500 vs MiniLM 0.833/0.347). The **reranker washes out the choice** — with 63 chunks, ~half the corpus enters the pool regardless of embedder. Keep MiniLM; switch to bge-small only if the reranker is dropped. |

> ⚠️ **The caveat that governs all of this: n = 6 questions.** One question moves hit@5 by 0.167.
> The **robust** signals are the hit@5 plateau, BM25 > dense, and the reranker washout. The fine MRR
> orderings are **within noise** — I would not re-tune production on them without 50–100 gold
> questions. The honest state is "measured, indicative, not yet conclusive".

Full tables, reproduce commands and per-sweep discussion: **[RETRIEVAL_EXPERIMENTS.md](RETRIEVAL_EXPERIMENTS.md)**.
