# Retrieval Experiments — measuring what I previously only reasoned about

Two fair criticisms prompted this phase:

1. *"How do you know which chunking would work well? You could have tried various chunking/embeddings."*
   — Correct. Phase 1's `chunk_size=1000, overlap=150` was chosen by **reasoning** (keep a slide's
   label next to its numbers; one page per chunk for clean citations), **not measurement**.
2. *"Are you doing contextual retrieval or HyDE?"* — **No**, in any phase. So I implemented and
   benchmarked both.

## Method (why these numbers are cheap and trustworthy)

Scoring an answer needs an LLM; scoring **retrieval** doesn't. So I built a gold set
(`src/experiments/goldset.py`) mapping each question → the page(s) that genuinely contain the answer,
**verified by locating the answer string in the deck** (not guessed). Several facts legitimately live
on 2–3 pages (total income is on p2 as a business breakdown *and* p22 as a Y-o-Y waterfall — both
correct), so a hit on **any** gold page counts.

- **hit@5** — did any gold page land in the top-5? (This is the metric that matters for us: we feed
  the top-5 to the LLM, and it picks.)
- **MRR** — 1/rank of the first gold page. (Measures ranking precision.)

> ⚠️ **Sample size: n=6 questions.** One question = 0.167 of hit@5. These results are *indicative,
> not conclusive* — a real evaluation needs 50–100+ questions. Stated up front so the numbers aren't
> over-read.

---

## 1. Does each retrieval layer earn its place?

| strategy | hit@5 | MRR |
|----------|------:|----:|
| sparse (BM25 only) | 0.833 | **0.833** |
| dense (FAISS only) | 0.833 | 0.347 |
| hybrid (RRF fusion) | 0.833 | 0.667 |
| **hybrid + rerank** (shipped) | **1.000** | 0.708 |

**The honest surprise: BM25 alone has the best *ranking* (MRR 0.833) — better than the full hybrid+rerank
stack (0.708).** And dense embeddings alone are weak (MRR 0.347): they find the right page *somewhere*
in the top-5 but rank it poorly. Fusing that noisy dense signal into BM25 actually **drags the ranking
down** (hybrid 0.667 < sparse 0.833).

Why: this is a **financial deck** — the discriminating tokens are exact strings (`H1-26`, `EBITDA`,
`IRM`, `44,281`). That's precisely BM25's strength and general-purpose embeddings' weakness.

**So was hybrid+rerank the right ship?** For *this* pipeline, yes — it's the only config with
**hit@5 = 1.000**, i.e. it never misses the gold page entirely, and hit@5 is what matters when you
hand 5 chunks to an LLM that then picks. But the nuance is real, and my Phase-1 narrative
("hybrid+rerank is best") was too clean. **BM25 is the workhorse here; the reranker buys coverage;
dense contributes the least.**

## 2. Chunking sweep

15 configs, `hybrid+rerank`, MiniLM. **Shipped config is marked ⬅.**

| chunk_size | overlap | #chunks | hit@5 | MRR |
|-----------:|--------:|--------:|------:|----:|
| 400 | 0 | 129 | 1.000 | 0.722 |
| 400 | 150 | 158 | 1.000 | 0.681 |
| 400 | 300 | 284 | 0.833 | 0.708 |
| 700 | 0 | 82 | 1.000 | 0.694 |
| 700 | 150 | 88 | 1.000 | 0.667 |
| 700 | 300 | 97 | 1.000 | 0.708 |
| 1000 | 0 | 63 | 1.000 | 0.792 |
| **1000** | **150** ⬅ shipped | 63 | 1.000 | **0.708** |
| **1000** | **300** | 68 | 1.000 | **0.806** ← best |
| 1500 | 0 | 49 | 1.000 | 0.792 |
| 1500 | 150 | 49 | 1.000 | 0.792 |
| 1500 | 300 | 49 | 0.833 | 0.750 |
| 2000 | 0/150/300 | 45 | 0.833 | 0.750 |

**What this actually says:**

- **My chosen `chunk_size=1000` was validated** — it sits in the 1000–1500 sweet spot where
  **hit@5 = 1.000**. The reasoning ("keep a slide's label with its numbers") held up.
- **My chosen `overlap=150` was NOT optimal** — it scores the *worst MRR in its own row* (0.708 vs
  0.806 @300 and 0.792 @0). So the tuning I *did* pick by intuition was the one that didn't pay off.
- **Chunk size is the lever that matters; overlap barely is.** At 2000 chars, hit@5 degrades to 0.833
  (chunks get coarse and blur multiple topics). Note `overlap` scarcely changes the chunk *count*
  (63→63 at size 1000) because most pages of this deck are smaller than the budget, so overlap rarely
  triggers — which is exactly why its effect is small and noisy.

> **Honesty on noise:** with n=6, an MRR delta of 0.08 ≈ half a rank position on a single question.
> The **robust** finding is the hit@5 plateau (1000–1500 good, 2000 bad). The overlap ranking is
> **within noise** — I would not re-tune production on it without a 50+ question set.

## 3. Embedding-model sweep

Three models, each with its **recommended query/passage prefix applied** (bge and e5 underperform
badly without them — fairness matters in a benchmark).

**With the full stack (hybrid + rerank) — all three are identical:**

| embedding model | hit@5 | MRR |
|-----------------|------:|----:|
| all-MiniLM-L6-v2 (shipped) | 1.000 | 0.708 |
| BAAI/bge-small-en-v1.5 | 1.000 | 0.708 |
| intfloat/e5-small-v2 | 1.000 | 0.708 |

Identical numbers are a **classic bug smell** (is the model even swapping?), so I checked instead of
reporting it. Running **dense-only** isolates the embeddings:

| embedding model | dense hit@5 | dense MRR |
|-----------------|------------:|----------:|
| all-MiniLM-L6-v2 (shipped) | 0.833 | 0.347 |
| **BAAI/bge-small-en-v1.5** | **1.000** | **0.500** |
| intfloat/e5-small-v2 | 0.667 | 0.556 |

**Not a bug — the models genuinely differ.** So the finding is real and more interesting:

> **The cross-encoder reranker completely washes out the embedding-model choice.**

Why: the fusion pool is top-15 dense + top-15 sparse out of only **63 chunks** — i.e. roughly half
the corpus enters the pool no matter which embedder you use. The reranker then re-scores the whole
pool and decides the final order. The embeddings only choose *who gets into the room*; the reranker
decides *the seating*.

**Consequences:**
- **Keep MiniLM** (smallest/fastest) — with a reranker, paying for a better embedder buys nothing here.
- **If you drop the reranker** (for latency/cost — see Phase 4B), **switch to bge-small**: it's clearly
  the best pure-dense retriever (hit@5 1.000 vs MiniLM's 0.833).
- ⚠️ **This will invert on a large corpus.** With 63 chunks the pool is half the corpus; with 63,000
  the pool is a rounding error and dense **recall** becomes decisive — then bge-small's advantage is
  the whole ballgame. This result is a property of *small-corpus + reranker*, not a universal law.

## 4. Contextual Retrieval (Anthropic's technique)

Each chunk is indexed with a short blurb situating it in the document, so an isolated block of
numbers stops being ambiguous.

| variant | hit@5 | MRR |
|---------|------:|----:|
| baseline | 1.000 | 0.708 |
| **contextual-cheap** (deterministic: doc title + page no.) | 1.000 | **0.792** |

**It works, and it's free** — MRR **0.708 → 0.792** with *no* LLM calls, just prepending
`[Context: AEL Q2 FY26 earnings presentation, page 22.]` to each chunk before embedding.

The full Anthropic method uses an **LLM to write each blurb** (`contextual-llm`, implemented in
`advanced.py`): that's **1 call per chunk = 63 calls** for this deck (~$0.008), and would likely beat
the deterministic version since the blurb could name the actual section. Left off by default to spare
the free-tier quota — the cheap variant already captures much of the win.

## 5. HyDE (Hypothetical Document Embeddings)

Instead of embedding the *question* (which looks nothing like the answer), ask the LLM to write a
*hypothetical answer* and embed **that** — it lives in the same "shape" of language as the real
passage. HyDE replaces only the **dense** probe; BM25 keeps the real question (its keywords matter).

| variant | hit@5 | MRR | cost |
|---------|------:|----:|------|
| baseline | 1.000 | 0.708 | — |
| **hyde** | 1.000 | **0.750** | +1 LLM call / query |

**HyDE helps (0.708 → 0.750) but less than contextual retrieval (0.792) — and unlike contextual, it
costs an LLM call on every query.** That tracks with §1: HyDE improves the *dense* probe, but dense is
the weakest signal on this corpus, so there's less to win. HyDE shines when queries and documents are
lexically dissimilar and embeddings carry the load — the opposite of a keyword-dense financial deck.

## Verdict

| technique | verdict here |
|-----------|--------------|
| BM25 (sparse) | **the workhorse** — best ranking on its own |
| Dense (MiniLM) | weakest link; contributes recall, hurts precision |
| Cross-encoder rerank | earns its place: only config with hit@5 = 1.000 |
| **Contextual retrieval (cheap)** | **best value — free, +0.084 MRR. Recommend adopting.** |
| HyDE | positive but small, and costs a call/query. Not worth it *here*. |
| Embedding model | **irrelevant while the reranker is on** — keep MiniLM. Switch to bge-small if the reranker is ever dropped. |

**Recommendation:** adopt cheap contextual retrieval; keep hybrid+rerank; don't adopt HyDE for this
corpus. Revisit HyDE if the corpus becomes prose-heavy (policies/contracts) where dense retrieval
carries more of the load.

**Reproduce:**
```bash
python -m src.experiments.bench strategies
python -m src.experiments.bench chunking
python -m src.experiments.bench embeddings
python -m src.experiments.advanced contextual   # free
python -m src.experiments.advanced hyde         # ~6 LLM calls
```
