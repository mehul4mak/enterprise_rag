# Model & Compute Strategy — Cost Research and Recommendation

**Context:** You offered ~$5–10 for a temporary GPU spin-up *or* an API key. This doc researches
both, sizes the actual workload, and justifies what we should use for Phase 1 (and how it feeds Phase 2).

> Prices below were checked live in **July 2026** (sources at the bottom). GPU spot prices drift daily;
> treat them as "order of magnitude", not quotes.

---

## 1. What this workload actually costs (the key insight)

A RAG agent sends the LLM only the **retrieved context**, not the whole PDF. Per question:

| Component | Tokens (approx) |
|-----------|-----------------|
| System prompt + grounding rules | ~350 |
| Top-5 retrieved chunks (~650 chars each) | ~900 |
| User question + chat history | ~250 |
| **Input total** | **~1,500** |
| Answer output | ~250 |

**Full acceptance suite** (5 required Qs + generalization on 2 more PDFs ≈ **30 questions/run**).
Even with **heavy dev iteration — say 500 LLM calls total across all debugging:**

- Input: 500 × 1,500 = 0.75M tokens → **$0.075** (at $0.10/1M)
- Output: 500 × 250 = 0.125M tokens → **$0.05** (at $0.40/1M)
- **Total ≈ $0.13 for 500 calls.**

➡️ **Your $5 credit = ~19,000 model calls.** For this task, API cost is effectively zero.
The whole "budget" conversation is really about *quality and alignment*, not spend.

---

## 2. Option A — Hosted API key (recommended for Phase 1 evaluation)

| Model | $/1M in | $/1M out | Free tier | Notes |
|-------|--------:|---------:|-----------|-------|
| **Gemini 2.5 Flash-Lite** ⭐ | $0.10 | $0.40 | ~1,000 req/day, no card | Cheapest active Google model; **directly on the Phase-2 GCP/Gemini path** |
| Gemini 2.5 Flash | ~$0.30 | ~$2.50 | shared free tier | Stronger reasoning if Flash-Lite underperforms on numerics |
| OpenAI gpt-4o-mini | $0.15 | $0.60 | none | Solid, provider-neutral |
| Claude Haiku 4.5 | low | low | none | Excellent instruction-following / refusal discipline |

**Recommendation: `gemini-2.5-flash-lite`.** Three reasons:
1. **Free tier (≈1,000 req/day) likely covers the entire demo at $0** — your $5 is a safety net you may never touch.
2. **Phase-2 alignment:** the JD is explicitly GCP + Gemini Enterprise + Gemini Agent Platform. Using Gemini now means Phase 2 is a *migration of the same calls to Vertex AI*, not a rewrite.
3. Quality is far above a local 2B model for grounded financial QA (numeric extraction, clean refusals).

**Already wired:** set `LLM_PROVIDER=gemini` + `GOOGLE_API_KEY=...` in `.env`. No code change needed.
Get a free key at https://aistudio.google.com/apikey.

---

## 3. Option B — Rent a GPU and run an open model (the "sovereign" experiment)

This is **more relevant than it first looks**, because the JD stresses *sovereign cloud* — data that
cannot leave the tenant. Running an **open-weight model on our own GPU** is the sovereign-aligned demo:
nothing is sent to a third-party API. Worth doing once as a Phase-2 credibility piece.

### What to rent (cheapest viable, July 2026 live rates)

| GPU | VRAM | RunPod Community | Vast.ai (marketplace) | Good for |
|-----|-----:|-----------------:|----------------------:|----------|
| **RTX 4090** ⭐ | 24 GB | ~$0.34/hr | ~$0.29–0.59/hr | 7B–14B instruct models, quantized |
| RTX 3090 | 24 GB | ~$0.22/hr | ~$0.15–0.25/hr | same, slightly slower |
| A40 | 48 GB | ~$0.39/hr | ~$0.35–0.45/hr | 14B–32B, longer context |

**$10 on a single RTX 4090 ≈ 29 hours** of runtime — far more than needed to run the full suite.

### What model to run on it (context is provided, so favor instruction-following over raw size)

| Model | Size | Fits on | Why |
|-------|-----:|---------|-----|
| **Qwen2.5-7B-Instruct** ⭐ | 7B | 24 GB easily | Strong at structured extraction + refusal; great numerics |
| Llama-3.1-8B-Instruct | 8B | 24 GB | Robust, well-known baseline |
| Qwen2.5-14B-Instruct | 14B | 24 GB (4-bit) / 48 GB | Noticeably better reasoning if 7B is borderline |

**How:** the spun-up box runs `ollama serve`; point our app at it by setting in `.env`:
```
LLM_PROVIDER=ollama
OLLAMA_HOST=http://<gpu-box-ip>:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```
**Zero code change** — the LLM layer already talks to any Ollama endpoint. We just change the host.

---

## 4. Decision

| Phase | Use | Cost | Rationale |
|-------|-----|-----:|-----------|
| **Phase 1 — default (offline proof)** | Local Ollama `gemma2:2b` on this machine | **$0** | Proves the pipeline runs fully offline, no key. Already working. |
| **Phase 1 — evaluation/quality** ⭐ | **Gemini 2.5 Flash-Lite API** | **~$0–0.15** | Best quality/effort ratio; likely free-tier; on the Phase-2 path |
| **Phase 2 — sovereign benchmark (optional)** | Rent 1× RTX 4090, run Qwen2.5-7B via Ollama | **~$3–10** | Demonstrates the sovereign / no-egress story the JD emphasizes |

**What I need from you to unlock the quality tier:** a free Gemini API key
(https://aistudio.google.com/apikey) dropped into `.env` as `GOOGLE_API_KEY` and `LLM_PROVIDER=gemini`.
Until then, everything runs and is testable on the local 2B model — I'll flag anywhere the small
model's quality (not the pipeline) is the limiting factor.

---

## 5. Stub status in code

- `src/llm.py` already implements **all four backends** (ollama / openai / anthropic / gemini).
- Switching is a `.env` change, no code edit.
- Default is `ollama` so the repo runs with **no key and no spend** out of the box.

### Sources
- [Runpod GPU pricing](https://www.runpod.io/pricing) · [Vast.ai pricing](https://vast.ai/pricing) · [Vast vs RunPod 4090 (Apr 2026)](https://www.synpixcloud.com/blog/vast-ai-vs-runpod-rtx-4090-pricing) · [RTX 4090 cloud comparison](https://getdeploying.com/gpus/nvidia-rtx-4090)
- [Gemini API pricing (official)](https://ai.google.dev/gemini-api/docs/pricing) · [Gemini pricing guide 2026](https://tokenmix.ai/blog/gemini-api-pricing)
