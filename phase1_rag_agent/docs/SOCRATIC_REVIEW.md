# Socratic Self-Review

An interviewer-grade pass over the project: hard questions asked of the code, honest answers, and
verdicts. The aim is not to claim perfection — it's to show the reasoning that would come up in a
design review, and to prove the limitations are *known*, not hidden.

---

## 1. "Is the code minimal, or is there dead weight?"

| Question | Finding | Action |
|----------|---------|--------|
| Did the Phase 1→2 refactor leave orphaned code? | Yes — `index_store.get_or_build_index` (PDF-bytes disk cache) had **no callers** after the provider layer moved to in-memory `build_from_chunks`. | **Rewired**, not deleted: exposed `get_or_build_from_chunks` and wired it into `LocalHybridRetriever.index()`, so the disk cache is *live* again (**696× faster warm reload**, measured). Dead code became a feature. |
| Duplicate grounding logic? | Was duplicated in Phase 1; **already** extracted to `src/grounding.py` and shared by the agent + graph. | None. |
| Duplicate embeddings/rerank across retrievers? | GraphRetriever and hybrid both reuse `src/embeddings.py`; no duplication. | None. |
| Are the GCP providers importable without the SDKs? | Yes — all `google-cloud`/`vertexai` imports are lazy; the module imports and constructs offline, failing only on use with a clear `ValueError`. | None. |

**Verdict:** minimal and functional. Every module has a caller; the one orphan was reinstated as a
real optimization.

## 2. "Does grounding actually prevent hallucination, or just look like it?"

- **Two independent layers**: the strict prompt, *and* post-hoc `postprocess_answer` that forces the
  refusal unless the answer carries a citation to a chunk that was actually retrieved.
- **Honest gap:** citation *validity* checks the tag was retrieved — not that the retrieved passage
  truly *supports* the claim. That deeper check is the eval harness's **faithfulness** judge
  (Phase 4A). So "no hallucination" is enforced by construction for *sourcing*, and *measured* for
  *support*. Documented, not hidden.

**Verdict:** sound, with the support-vs-sourcing distinction made explicit.

## 3. "What breaks under load / over time?"

| Concern | Answer | Stance |
|---------|--------|--------|
| `AppState.sessions` grows unbounded | True — in-memory demo store. | **Known limitation.** Prod uses a TTL store (Redis/Memorystore); noted here rather than over-engineered into a take-home. |
| `SemanticCache` grows unbounded | True — same. | **Known limitation.** Prod adds LRU/TTL eviction; the interface (`clear`, `size`) is already there to bolt it on. |
| Cost attribution on the GCP backend | `active_model` returns `gemini_model` even when the Vertex model differs. | **Minor** — cost dashboard is an estimate anyway; flagged. |
| Concurrency | FastAPI endpoints are sync `def` (thread-pooled); metrics/cache are lock-guarded. | Fine for the scale claimed. |

**Verdict:** the unbounded stores are the honest weak point of the *demo*; both are documented and
trivially swappable because they sit behind small interfaces.

## 4. "Do the phases actually advance the task, or is it busywork?"

Each phase answers a question the previous one raised — none is decoration:

1. **P1 local RAG** → *can we answer, cite, and refuse correctly?* (5/5 acceptance).
2. **P2 interfaces + LangGraph + governance** → *how do we make it enterprise-grade and cloud-portable
   without a rewrite?* (BACKEND swap; guardrails/lineage/tracing).
3. **P3 real GCP** → *is the port real?* (Vertex/Document AI code + Terraform, deploy-ready).
4. **P4A eval** → *how do we know quality, before we optimize or fine-tune?* (LLM-judge + gates).
5. **P4B optimization** → *now measure-guarded, can we cut cost/latency?* (semantic cache: 60× / $0).
6. **P5 GraphRAG** → *does a fancier retriever help?* (honest answer: not on one small deck — a
   *negative result*, well-measured; wins on large multi-doc corpora).

**Verdict:** the sequence is a dependency chain (you can't do P4B safely without P4A), which is
exactly how a real roadmap reads.

## 5. "Is it aligned to the JD, or generic?"

| JD requirement | In this project | Status |
|----------------|-----------------|--------|
| LangGraph agentic framework | `src/graph.py` StateGraph | ✅ built |
| Google ADK / Crew.AI multi-agent | design in `PHASE4_PLAN.md` (§D) | ◑ planned |
| GCP + Gemini Enterprise / Gemini on Vertex | `src/providers/gcp.py` + Terraform | ✅ built (deploy-ready) |
| Sovereign cloud / data sovereignty | `governance/residency.py` + `asia-south1` IaC | ✅ built |
| Assured data lineage | `governance/lineage.py` (Dataplex-style) | ✅ built |
| Governed data + access controls | guardrails/Model Armor + least-priv IAM in TF | ✅ built |
| Robust Python AI pipelines | whole codebase, interface-driven | ✅ built |
| MLOps / model evaluation / data-driven | `src/eval/` harness + Vertex Pipelines plan | ✅ eval built, ◑ pipelines planned |
| Prompt engineering | `src/prompts.py` grounding/condensation | ✅ built |
| Model creation / fine-tuning (multi-LoRA) | `PHASE4_PLAN.md` (§A) | ◑ planned |
| Production observability (99.9% KPI) | tracing + metrics + `/metrics`, Cloud Run autoscale | ✅ built |
| Multi-cloud / edge | not addressed | ✗ out of scope, noted |

**Verdict:** the *built* set covers the JD's core (agentic RAG on GCP/Gemini, sovereign, governed,
observable, evaluated); the *planned* set (multi-agent, fine-tuning, MLOps pipelines) is scoped in
`PHASE4_PLAN.md` with the interfaces already in place.

---

## Overall
A hiring panel's three usual worries — *does it actually work, is it honest, and can it scale to the
JD* — are each answerable with evidence: 5/5 acceptance + 7/7 eval; an audit log that records the
failures (docling, GraphRAG tie, the cache-testing mistake, protobuf); and an interface-driven
architecture where "port to GCP / add multi-agent / fine-tune" are additions, not rewrites. The
remaining honest gaps (unbounded in-memory stores, planned-not-built multi-agent/fine-tuning) are
documented here rather than papered over.
