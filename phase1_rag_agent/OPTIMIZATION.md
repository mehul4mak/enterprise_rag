# Phase 4B — Latency & Cost Optimization

Now that the eval harness (Phase 4A) can prove quality doesn't regress, we optimize. Two things
shipped this phase: **per-stage cost instrumentation** (so we can *see* cost) and a **semantic answer
cache** (the biggest latency/cost win). Everything is gated on "quality must not drop."

---

## 1. Cost + token instrumentation

Every LLM stage (`condense`, `generate`) now records estimated **input/output tokens** and **cost_usd**
on its trace span and into metrics (`src/observability/cost.py`).

- Tokens are estimated (~4 chars/token) since not every backend returns exact usage — documented as an
  estimate, not a bill. On Vertex you'd swap in the returned `usage_metadata`.
- Pricing table per model (e.g. `gemini-flash-lite-latest` = $0.10 / $0.40 per 1M in/out; local = $0).
- Surfaced per request in `/chat` (`cost_usd`) and in aggregate at `/metrics`
  (`llm.input_tokens`, `llm.output_tokens`, `llm.cost_micro_usd`).

Example: one grounded answer ≈ **$0.00013** (1.3 hundredths of a cent). This is what makes the
per-query cost visible for SLA/budget decisions.

GCP mapping: Cloud Billing export + per-request cost labels; Cloud Monitoring for the aggregates.

## 2. Semantic answer cache (the marquee win)

`src/optimize/semantic_cache.py` — embed the query (cheap, local) and, if a previously answered query
is within a cosine `CACHE_THRESHOLD` (default 0.97), return the cached answer, **skipping retrieval +
generation entirely**.

Enable with `SEMANTIC_CACHE=on`.

### Measured result (cross-session, same document)
| | latency | cost | cache_hit |
|--|--------:|-----:|:--------:|
| user 1 — fresh question | 8.05 s | $0.00013 | ❌ (miss) |
| user 2 — same question, **paraphrased** | **0.13 s** | **$0** | ✅ (hit) |

**≈60× faster, 100% cost saved** — and it matched a *paraphrase* ("...income **for** H1-26" hit the
"...income **in** H1-26" cache; their embedding cosine was 0.9935). That's the point of a *semantic*
(not exact-string) cache.

### Correctness rules (important)
- **Per-document keying** — answers never leak across documents (unit-tested).
- **Only standalone (no-history) questions are cached.** A follow-up like "break that down" depends on
  conversation history, so serving it from a query-only cache would be wrong. Consequence: within a
  single chat session, only the *first* question is cacheable; the real payoff is **cross-session**
  (FAQ-style repeats from different users hitting one shared cache).
- Blocked (guardrail) answers are never cached.

### Threshold tuning (honest note)
0.97 is deliberately tight to avoid serving a cached answer to a *different* question. It still
catches close paraphrases (0.9935 above), but rewordings that drop below 0.97 will (correctly) miss
and regenerate. Tune `CACHE_THRESHOLD` per risk tolerance; a stronger embedding model widens the safe
band.

GCP mapping: a managed semantic cache on Memorystore / Vertex — identical pattern.

## No-regression argument
The cache returns **byte-identical** stored answers (verified `same_answer=True`), and cost tracking
is passive (doesn't touch the answer path). The 7 eval questions are all distinct → every one is a
cache *miss* → the exact Phase-4A pipeline, which scored **7/7 all-gates-pass**. So quality is
unchanged by construction; the eval harness remains the gate for any future, less-passive optimization
(model tiering, prompt compression).

## Deferred (next optimization levers)
- **Model tiering** — a cheaper/smaller model for `condense` vs the final answer. On
  `gemini-flash-lite` (already the cheapest tier) the gain is marginal; matters most with a big/small
  split. Needs a per-call model override on `LLMProvider`.
- **Embedding cache**, **prompt/context compression**, **adaptive top-k**, **async stages**. Each is
  guarded by the eval harness.
