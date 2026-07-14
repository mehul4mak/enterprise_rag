# Phase 4 — Plan (beyond RAG: adapters, optimization, PII vault, non-RAG GenAI)

Phase 4 moves from "a good RAG service" to "a governed GenAI platform." Four workstreams, each
mapped to GCP and to the JD (which names ADK, CrewAI, LangGraph, fine-tuning, MLOps, data lake,
sovereign cloud). Nothing here is built yet — this is the design we'd execute.

---

## A. Multi-LoRA adapters + fine-tuning

**Goal:** specialize a shared base model cheaply with several swappable LoRA adapters, instead of one
monolithic fine-tune.

- **Adapters:** (1) *financial-QA grounding* style, (2) *strict citation formatting*, (3)
  *refusal/confidence calibration*, (4) a *guardrail classifier* (replaces the regex in Model Armor).
- **Serving:** one base model + hot-swappable adapters. GCP: Vertex AI **multi-LoRA** endpoints;
  local: **vLLM/LoRAX** or Ollama adapter files. A tiny **router** classifies each query → adapter.
- **Data flywheel:** training pairs bootstrap from the **lineage logs we already write**
  (`question → retrieved → answer → cited`) plus synthetic Q/A generated from documents. SFT → DPO
  on preference pairs (good vs hallucinated/uncited answers).
- **MLOps:** **Vertex AI Pipelines** for train → eval-gate → **Model Registry** → canary deploy;
  data-driven retraining triggered when eval metrics or lineage drift cross thresholds.
- **Interfaces:** add `Adapter`/`Router` seams next to the existing `LLMProvider`; the graph gains a
  `route` node before `generate`. No change to retrieval/governance.

## B. Per-step latency + cost optimization

**Goal:** hit an SLA (e.g. p95 < 2s, cost < ₹X/query) by optimizing each stage. We already emit
per-stage **latency**; add a per-stage **cost** dimension (tokens × price).

- **Instrumentation:** extend `observability/` with a `cost` span attribute + a cost metric; a
  budget/SLA report per query and aggregate.
- **Levers:**
  - **Model tiering** — tiny/cheap model for `condense` + routing, big model only for final synthesis.
  - **Semantic answer cache** (embedding-keyed) + **embedding cache** for repeated/near-duplicate queries.
  - **Batched** embeddings, **async** I/O across independent stages, **prompt compression**, adaptive
    `top_k` (fewer chunks when confidence is high).
  - **Quantized** local models; **slim the GCP image** (lazy-import sentence-transformers so the
    Vertex backend doesn't ship unused ML deps — flagged during Phase 3).
- **Autotuning:** an optimizer that picks knobs per query class to meet the SLA (quality floor via the
  eval harness in D).

## C. PII dual-store (masked + raw) — tokenization vault

**Goal:** the working system only ever sees masked/tokenized text; raw PII lives in a separate,
tightly controlled vault. (Directly your idea: two DBs.)

```
ingest ─▶ DLP detect PII ─▶ replace with reversible tokens ─▶ WORKING STORE (index, logs, RAG)
                          └▶ raw value ──────────────────────▶ RAW VAULT (KMS/CMEK, tight IAM)
answer ─▶ contains a token? ─▶ re-identify ONLY if caller authorized (IAM) ─▶ audit every re-id
```

- **Working store:** existing index + `.cache`/BigQuery — tokenized text only.
- **Raw vault:** **KMS/CMEK-encrypted** Firestore/BigQuery (the Terraform already provisions the KMS
  key `pii_vault`), separate IAM, no analyst read access.
- **Tokenization:** GCP **DLP** format-preserving/deterministic tokens (reversible with a wrapped key)
  — deterministic so the same entity maps to the same token (joinable) without exposing the value.
- **Re-identification:** a narrow, audited service that swaps tokens→raw only for authorized flows;
  every call logged to the lineage/audit sink. Upgrades the Phase-2 regex redaction to a
  vault-backed, reversible, access-controlled scheme.

## D. Non-RAG GenAI (touching the wider platform)

- **Structured extraction:** deck → normalized **financial JSON** via schema-constrained decoding /
  function-calling (feeds analytics, not just chat).
- **Text-to-SQL analytics agent** over the **Enterprise Data Lake** (JD) — NL → BigQuery with
  guardrails + result grounding.
- **Multi-agent orchestration** with **Google ADK + CrewAI** (both named in the JD): a supervisor
  delegating to a retriever-agent, a **calculator/tool-use agent** (derived metrics, YoY deltas), and
  the SQL agent — LangGraph remains the in-process engine, ADK/CrewAI for cross-agent coordination.
- **Summarization & cross-document/quarter comparison** pipelines.
- **Evaluation harness:** LLM-as-judge + groundedness/faithfulness/citation-accuracy metrics as **CI
  regression gates** (also the quality floor for B's optimizer and A's retrain triggers).

---

## Suggested sequencing
1. **D-eval harness first** (gives an objective quality gate everything else needs).
2. **B optimization** (cheap wins; instrumentation already half-built).
3. **C PII vault** (governance-critical; KMS key already provisioned).
4. **A adapters + fine-tuning** (biggest lift; needs the eval harness + data flywheel).

Each lands on its own branch (`phase-4-*`) with the same rigor: interfaces, tests, audit log, and
GCP-parity docs.
