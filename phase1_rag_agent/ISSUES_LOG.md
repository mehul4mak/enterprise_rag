# Engineering Audit Log — Phase 1 RAG Agent

A running, honest record of every issue encountered during the build: setup, dependency, API, logic, framework, and result-quality problems — and how each was diagnosed and resolved. Newest entries appended at the bottom of each section.

Legend for **Status**: ✅ Resolved · 🔶 Worked around · 🔴 Open · ℹ️ Observation/decision (not a bug)

---

## 0. Environment discovery (baseline facts)

| Item | Finding |
|------|---------|
| Python | 3.12.4 |
| RAM | 7.6 GiB total (~5 GiB free) — **constrains local model choice** |
| GPU | None (`nvidia-smi` not found) — CPU-only inference |
| CPU | 12 cores |
| Internet | Available (pypi 200) |
| API keys | **None set** for any provider (OpenAI/Anthropic/Google all empty) |
| Ollama | Installed, server live on :11434, 12 local models present |
| Pre-installed libs | langchain stack, faiss-cpu, chromadb, sentence-transformers, transformers, pypdf, tiktoken |

**Decision (ℹ️):** User chose **local open-source model** (no API cost/key) + **.env** for optional keys. This forces the whole pipeline to run offline, which is a stronger portability guarantee anyway.

---

## 1. Installation & dependency issues

### 1.1 ℹ️ Missing PDF/BM25 libraries
- **Context:** `pip list` showed `pypdf` but not `pymupdf`, and no `rank_bm25`.
- **Why it matters:** `pypdf` gives weaker layout/text extraction than PyMuPDF (`fitz`); BM25 is needed for the bonus hybrid retrieval.
- **Action:** `pip install pymupdf rank_bm25` — installed cleanly, no conflicts.
- **Status:** ✅ Resolved.

### 1.2 ℹ️ pip version-notice noise
- **Observation:** Every `pip install` prints a "new release of pip available" notice to stderr. Cosmetic only; ignored.
- **Status:** ℹ️ Observation.

---

## 2. Document / source acquisition issues

### 2.1 ℹ️ Sample PDF links were not in the visible doc text
- **Context:** The task `.docx` shows "Sample 1 / Sample 2" as labels but the actual URLs were embedded as Word hyperlink relationships, not inline text.
- **Action:** Extracted the `.docx` relationship XML (`doc.part.rels`) to recover all 4 URLs (primary earnings deck + press release + board outcome + a duplicate of the deck).
- **Result:** Downloaded 3 distinct PDFs (earnings 41pp, press release 4pp, board outcome). All valid PDF v1.7.
- **Status:** ✅ Resolved.

---

## 3. Ingestion & chunking issues

### 3.1 🔶 PyMuPDF table detection is unreliable on this deck
- **Context:** The financial figures live in slide-style layouts (e.g., page 2 "Total Income / EBITDA / PBT" with numbers scattered as separate text spans). `page.find_tables()` returned **0 tables** on page 2 despite obvious tabular data, and emitted a hint to use the paid `pymupdf_layout` package.
- **Impact:** Cannot rely on structured table extraction. Numbers like `49,263 / 44,281` come through as loose lines in reading order.
- **Workaround:** Use plain `get_text("text")` per page and lean on **chunking that keeps a whole slide's text together** so the label ("Total Income") and its values stay in the same chunk. Verified page 2 chunk `c2` contains both the labels and `49,263 / 44,281` together — so retrieval + LLM can still associate them.
- **Status:** 🔶 Worked around (no paid dependency). Noted as a Phase-2 improvement candidate (layout-aware parsing / Document AI).

### 3.2 ℹ️ Header/footer noise ("Hamburger Menu Icon", "STRICTLY CONFIDENTIAL")
- **Observation:** Slide chrome repeats on many pages and pollutes chunks.
- **Decision:** Left in for now (low harm; reranker down-weights it). Candidate for a stopword/boilerplate filter if it hurts retrieval precision. Will revisit after acceptance tests.
- **Status:** ℹ️ Observation / deferred.

---

## 4. Indexing issues

### 4.1 ℹ️ Cold build latency dominated by model load
- **Observation:** First index build took ~16 s; cached reload ~0.02 s. The 16 s is almost entirely `sentence-transformers` model load + first-batch encode on CPU, not the 63-chunk embed itself.
- **Decision:** Added content-hash-keyed disk cache (`.cache/<md5>`) so re-runs on the same PDF are instant. Cache key includes chunk params + embedding model name so changing config invalidates correctly.
- **Status:** ✅ Resolved (by design).

---

## 5. Retrieval quality issues

### 5.1 ℹ️ RRF fused score alone under-ranks the best chunk; reranker fixes it
- **Context:** For "consolidated total income in H1-26", pure RRF put `[p23:c36]` (Q2-26 income) and `[p17:c27]` (airport cargo) near the top — semantically close but wrong period/section.
- **Fix that worked:** The cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) re-scored candidates and pushed the two *correct* chunks to rank 1–2: `[p22:c35]` (H1-26 TOTAL INCOME) rerank=4.58 and `[p2:c2]` (consolidated highlights, contains `49,263 / 44,281`) rerank=4.57 — clearly separated from the next best (0.19).
- **Takeaway:** Hybrid + rerank is pulling its weight; RRF alone would have been mediocre here. Validates the "Excellent (bonus)" reranking requirement.
- **Status:** ✅ Working as intended.

### 5.2 ℹ️ Header/footer noise did NOT block retrieval
- The boilerplate ("Hamburger Menu Icon", "STRICTLY CONFIDENTIAL") is present in top chunks but the reranker still scored them correctly. Boilerplate stripping deferred as low-priority.
- **Status:** ℹ️ Observation.

---

## 5b. Environment / tooling issues

### 5b.1 ✅ Two Python installs → `uvicorn` console script used the wrong one
- **Symptom:** `uvicorn src.api:app` crashed at startup with `ModuleNotFoundError: No module named 'faiss'`, even though `python3 -c "import faiss"` works fine.
- **Root cause:** This box has **two Pythons** — miniconda **3.12** (where all deps are installed, and where `python3` points) and a system **3.10** with `~/.local` packages. The bare `uvicorn` console script has a shebang to the 3.10 interpreter, which lacks faiss/sentence-transformers.
- **Fix:** Launch via the module form — **`python3 -m uvicorn src.api:app`** — which uses the same 3.12 interpreter as everything else. Verified: `/health`, `/ingest`, and a real `/chat` all returned correctly (answer `₹ 44,281 crore [p22:c35],[p2:c2]`).
- **Lesson:** On multi-Python machines, prefer `python -m <tool>` over bare console scripts. (In the Docker image there's a single Python, so `uvicorn ...` is fine there.)
- **Status:** ✅ Resolved.

### 5b.2 ✅ End-to-end hosted path verified
- Started the FastAPI service, ingested the deck (63 chunks/41 pages), and ran a live `/chat` → correct grounded answer + citations + top-k retrieval debug in the JSON response. The hosting deliverable is proven, not just wired.
- **Status:** ✅ Verified.

## 6. LLM / generation issues

### 6.1 ✅ Local Ollama path verified
- Smoke test: `generate("Reply with exactly the word: OK", ...)` via `gemma2:2b` returned `'OK'`. Local, offline, zero-cost generation path confirmed working before wiring the RAG prompt.
- **Status:** ✅ Working.

### 6.2 ⚠️→✅ Default Gemini model was already deprecated
- **Caught during cost research:** I had defaulted `GEMINI_MODEL=gemini-2.0-flash`. Web check (July 2026) showed **Gemini 2.0 Flash / Flash-Lite were shut down 2026-06-01**. Had we shipped that default, the Gemini path would 404 at runtime.
- **Fix:** Changed default to `gemini-2.5-flash-lite` (cheapest active model, $0.10/$0.40 per 1M) in both `config.py` and `.env.example`, with an inline comment noting the deprecation.
- **Lesson:** Never hardcode a model id from memory — verify it's live. (This is exactly why the repo pins model ids in one config file, not scattered in code.)
- **Status:** ✅ Resolved.

### 6.4 🔴→🔶 Local Ollama timed out (>180s) on first acceptance question
- **Symptom:** The unbuffered acceptance rerun crashed on question 1 ("major business segments") with `TimeoutError: timed out` from the Ollama HTTP call (180s limit) → wrapped as `LLMError`.
- **Analysis:** The single numeric question earlier answered in ~90s, but under a fresh subprocess (cold model load + a broader "list all segments" prompt that pulls more context) `gemma2:2b` on a 12-core CPU with ~5 GiB free RAM exceeded 180s. This is a **CPU/RAM constraint**, not a logic bug.
- **Response:** (a) It motivated switching to the just-provided Gemini key for the eval run; (b) local remains a valid offline proof but is slow/flaky for full suites on this box. Would bump the timeout or use a smaller/faster model for a pure-local demo.
- **Status:** 🔶 Worked around (moved eval to Gemini; local still works for single questions).

### 6.5 🔴→✅ Gemini free-tier: key format, model mapping, and 429 rate limits
- **Key format:** the provided key is the **new `AQ.` format** (not `AIza`). It authenticates fine with `google-generativeai` (`genai.configure`) — confirmed by getting a 429 (quota) rather than 401 (auth).
- **Model gotcha:** `gemini-flash-latest` resolves to **`gemini-3.5-flash`**, whose free tier allows only **5 requests/minute** — a 6–7 call suite blows it instantly. Got 429 on the first call (shared key likely already used by the team).
- **Fixes applied:**
  1. Pinned `GEMINI_MODEL=gemini-2.5-flash-lite` (higher free quota than 3.5-flash).
  2. Added **429 backoff/retry** (0/20/40/60s) in `_gemini_generate` respecting the free-tier limit.
  3. Will space out acceptance calls and run the suite **once** (user said "use sparingly").
- **Privacy note (from the key's README):** free tier may train on submitted content → *do not send confidential Adani data*. Our earnings deck is **public investor material**, so it's acceptable; flagged in docs for anyone reusing this with private docs.
- **Security:** key lives only in gitignored `.env`; never committed.
- **Status:** ✅ Resolved (hardened + documented).

### 6.6 🔴→✅ Gemini model 404s: new keys only get the newest models
- **Symptom:** After pinning `gemini-2.5-flash-lite`, the suite crashed with `404 This model ... is no longer available to new users`. Same for `gemini-2.5-flash`.
- **Diagnosis:** Listed the key's models via REST (`GET /v1beta/models`) — a **secondary bug surfaced**: the SDK's `list_models()` threw `AttributeError: FieldDescriptor object has no attribute 'is_repeated'` (protobuf-version mismatch in the installed `google-generativeai`), so I fell back to raw `curl` to enumerate. The 2.5 models appear in the list but 404 on `generateContent` — because this is a **newly-created key/project**, which Google restricts to the **newest 3.x models** only. Grandfathered projects keep 2.x.
- **Fix:** Use the version-agnostic **`*-latest` aliases**. `gemini-flash-lite-latest` returned `'OK'` on a single call. Set it as default in `config.py` + `.env.example` with an explanatory comment.
- **Lesson:** For shared/unknown keys, prefer `*-latest` aliases over pinned version numbers.
- **Status:** ✅ Resolved.

### 6.7 ℹ️ Added inter-call spacing for hosted free tier
- Free-tier RPM is tight (`gemini-flash-latest`/3.5-flash = 5 RPM). Added 5s spacing between acceptance questions when `provider=gemini`, on top of the 429 backoff, so a 6–7 call suite stays under the limit.
- **Status:** ℹ️ Mitigation.

### 6.3 ℹ️ Cost sizing → API is effectively free for this workload
- Sized the acceptance suite at ~1,500 input + 250 output tokens/question. 500 dev calls ≈ **$0.13**. Full analysis + GPU-rental alternative in `COST_AND_MODEL_RESEARCH.md`.
- **Decision:** default stays local `gemma2:2b` ($0, offline proof); recommend Gemini `*-lite-latest` for the quality/eval tier (aligns with Phase-2 GCP/Gemini).
- **Status:** ℹ️ Decision recorded.

---

## 7. Result-quality / acceptance-test issues

### 7.0 ✅ Tooling gotcha: Python block-buffers stdout to a file → blind during long runs
- **Symptom:** Ran the acceptance suite with output redirected to a file; the file stayed **0 bytes** for minutes even though the process was alive at ~21% CPU with `gemma2:2b` loaded. A `Monitor` grepping for the end marker never fired.
- **Root cause:** CPython uses **block buffering** (not line buffering) when stdout is a pipe/file, so nothing is written until the buffer fills or the process exits. On a slow CPU run (~6–7 LLM calls) that means total blindness until the very end.
- **Fix:** Re-run with `python3 -u` (or `PYTHONUNBUFFERED=1`) for live, per-question output; point the monitor at the actual output file.
- **Lesson:** Always run long background Python with `-u` when you need progress visibility.
- **Status:** ✅ Resolved.

### 7.2 ✅ Full acceptance suite: 5/5 passed on Gemini
- Ran the mandated 5 scenarios end-to-end on `gemini-flash-lite-latest`: grounded-fact, numeric, cross-section, negative-control, and multi-turn follow-up — **all PASS** (transcript in `tests/acceptance_results_gemini.txt`).
- **Follow-up correctness proof:** "Break that down into passenger and cargo changes" (no explicit subject) was condensed into a standalone query and retrieved `[p16:c26]`: Pax 45.1→46.0 Mn (+4%), Cargo 5.5→5.7 L-MT (+4%). Chat-history awareness works.
- **Negative control proof:** "CEO's email" → "Not found in the document." even though 5 chunks were retrieved — the grounding/citation-validation refused rather than fabricated.
- Per-question latency 3.6–9.6s on Gemini (vs ~90s local). No 429s with model=`*-lite-latest` + 5s spacing + backoff.
- **Status:** ✅ Pass.

### 7.1 ✅ Numeric question passed on the local 2B model (better than expected)
- **Q:** "What is the consolidated total income in H1-26?"
- **A:** `44,281 ₹ crore [p22:c35]` — correct value, correct period (H1-26 not Q2), valid citation to a retrieved chunk.
- **Why notable:** I expected a 2B model to fumble numeric extraction from noisy slide text. It didn't, because retrieval+rerank handed it the *right* chunk and the strict prompt forced verbatim copying. This validates the "retrieval does the heavy lifting, LLM just extracts+cites" design — small model is viable for the pipeline proof.
- **Status:** ✅ Pass.

---

# ===== PHASE 2 (GCP-shaped, running locally) =====

## 8. Framework / dependency issues (Phase 2)

### 8.1 🔴→🔶 `langchain-google-genai` / `-vertexai` won't import (protobuf skew)
- **Symptom:** `import langchain_google_genai` (and `langchain_google_vertexai`) crash with
  `AttributeError: 'google.protobuf.pyext._message.FieldDescriptor' object has no attribute 'is_repeated'`.
- **Root cause:** Same defect as Phase 1 §6.6. Installed **protobuf 6.33.5** has a C-extension /
  pure-Python skew — `json_format.py` calls `field.is_repeated` but the compiled `_message`
  FieldDescriptor doesn't expose it. Any code path that serializes a proto to dict at import time
  (which these LangChain-Google packages do) blows up.
- **Why I did NOT just fix protobuf:** this is a **shared machine** running the user's other
  services (odoo, open-webui, a local coding-agent) and many `google-cloud-*` libs pin protobuf
  ranges. Downgrading/upgrading protobuf globally risks breaking unrelated software. Not my call to
  make on a shared env.
- **Workaround (chosen):** Build Phase 2 on **LangGraph + my own thin provider wrappers** over the
  *working* raw `google-generativeai` `generate_content` path (that path never hits the broken
  json_format code — proven in Phase 1). No dependency on the LangChain-Google packages.
- **Phase 3 note:** the cloud image will pin a matched protobuf (e.g. the version
  `google-cloud-aiplatform` requests), at which point `langchain-google-vertexai` becomes usable if
  desired. Documented in `src/providers/gcp.py`.
- **Status:** 🔶 Worked around (LangGraph without langchain-google); root cause documented.

### 8.2 ℹ️ Architecture decision: interfaces + factory, not a rewrite
- **Goal of Phase 2** ("GCP-shaped but local"): make Phase 3 a *config swap*, not a rewrite.
- **What I built:**
  - `src/providers/base.py` — 4 interfaces (DocumentParser, Embedder, Retriever, LLMProvider),
    each mapping 1:1 to a GCP managed service.
  - `src/providers/local.py` — local impls wrapping the proven Phase 1 code.
  - `src/providers/gcp.py` — Vertex/Document AI **stubs** with the exact SDK entry points sketched;
    they raise a clear "Phase-3, needs GCP creds" error until wired.
  - `src/providers/factory.py` — selects impls from `BACKEND=local|gcp`.
  - `src/graph.py` — the agent rebuilt as a **LangGraph StateGraph** (condense→retrieve→generate→
    validate) depending only on the interfaces.
  - `src/grounding.py` — extracted the citation-validation/refusal logic so Phase 1 agent and the
    graph share ONE implementation (DRY).
- **Verification:** local backend gives identical grounded answers; multi-turn follow-up condensed
  and retrieved `[p16:c26]` (Pax 45.1→46.0, Cargo 5.5→5.7) through the graph. `BACKEND=gcp` builds
  the stubs and raises `NotImplementedError` as designed (unit-tested).
- **Status:** ℹ️ Decision recorded; implemented + tested.

### 8.3 ✅ LangGraph 1.0.1 works cleanly
- Confirmed `StateGraph` compiles and runs (trivial graph + the real 4-node RAG graph). No proto
  issues — LangGraph core doesn't touch the broken google-proto paths.
- **Status:** ✅ Verified.

---

# ===== PHASE 2+ (governance, safety, observability, multi-parser) =====

## 9. Git hygiene

### 9.1 ✅ Removed Claude co-author trailer from all commits (user request)
- **Ask:** user did not want `Co-Authored-By: Claude …` on commits.
- **Action:** rewrote all 3 branches with `git filter-branch --msg-filter` (a Python script stripping
  the trailer + trailing blank lines), removed `refs/original/` backups, force-pushed `main`,
  `phase-1-local-rag`, `phase-2-gcp-local`. Verified author/committer identity stays the user's and
  0 trailers remain. Won't add it going forward.
- **Status:** ✅ Resolved.

## 10. Governance / safety / observability (Phase 2+)

### 10.1 ℹ️ Built the full GCP-parity governance layer locally
- Implemented, each mapping 1:1 to a GCP service (documented in GOVERNANCE.md):
  guardrails + **Model Armor** (injection screen + PII redaction), **data residency** enforcement
  (strict = local-only), **lineage** (Dataplex-style JSONL), **tracing** (Cloud Trace-style per-stage
  spans + latency), **metrics** (Cloud Monitoring-style), structured **logging** (Cloud Logging-style).
- Integrated into the LangGraph graph as `input_guard`/`output_guard` nodes + per-node trace spans +
  a `finalize` node (lineage/metrics/log). API surfaces per-stage `latency_ms`, `trace_id`,
  `guard_findings`; added `GET /metrics`.
- **Verified:** injection → blocked pre-LLM; normal query → grounded + lineage recorded; residency
  strict blocks gemini; PII redacted. 10 governance unit tests green (27 offline tests total).
- **Status:** ℹ️ Implemented + tested.

### 10.2 🔴 `unstructured` PDF parser needs heavy extra deps
- `unstructured.partition.pdf` import fails: first `pi_heif` (installed), then `unstructured_inference`
  (pulls detectron2-class layout models — heavy). Deferred rather than bloat the env.
- **Workaround:** compared 5 other methods instead (PyMuPDF, pdfplumber, docling, EasyOCR, moondream
  VLM). `unstructured` documented as an optional Phase-3/Document-AI-adjacent path.
- **Status:** 🔴 Deferred (documented).

## 11. Visual-PDF parsing comparison (Phase 2+)

### 11.1 ℹ️ Benchmarked 5 methods — pdfplumber won, docling surprisingly lost the numbers
- Tried **pymupdf, pdfplumber, docling, easyocr, moondream-VLM** on the earnings deck; scored on
  whether "Total Income" ends up next to its value (44,281 / 49,263). Full table in PARSING.md.
- **Findings:**
  - `pdfplumber` — only full-doc method keeping label↔number adjacent (13 s). **Best here.**
  - `pymupdf` — has the numbers but not adjacent (the Phase 1 limitation, quantified).
  - `docling` — **223 s and LOST 44,281/49,263 entirely.** Its table/layout model expects real
    tables; these infographic slides (numbers as positioned text) defeated it. "Fancier ≠ better."
  - `easyocr` — render→OCR recovers numbers + adjacency (visual read) but per-page slow; right for
    **scanned** docs, overkill here.
  - `moondream VLM` — got the *labels* but **not the numbers** (69 s/page). A 1.7B VLM is too small
    for dense financial figures; would need Gemini-vision or a 7B+ VLM.
- **Action:** added `PARSER=pymupdf|pdfplumber|docling|easyocr|vlm` env selection (default pymupdf;
  pdfplumber recommended for numeric locality). Heavy deps isolated in `requirements-parsers.txt`.
- **Status:** ℹ️ Explored, documented, wired.

---

# ===== PHASE 3 (real GCP — deploy-ready, no spend) =====

## 12. GCP implementation

### 12.1 ℹ️ No GCP creds locally → deploy-ready code + IaC (user-approved)
- No `gcloud`/ADC in the dev env; real provisioning costs money. Per user decision: build
  **deploy-ready code + Terraform IaC + setup guide, no spend**; target **asia-south1 (Mumbai)** for
  India sovereignty.
- **Delivered:** real `src/providers/gcp.py` bodies (Document AI, Vertex embeddings, Vertex Vector
  Search + Ranking API, Gemini-on-Vertex); `infra/terraform/` (APIs, GCS, Doc AI processor, Vector
  Search index+endpoint+deploy, Artifact Registry, KMS PII-vault key, runtime SA + least-priv IAM,
  Cloud Run); `cloudbuild.yaml` + `Dockerfile.gcp` + `requirements-gcp.txt`; `GCP_SETUP.md`.
- **Honesty:** the GCP provider code + IaC are written against current SDK/provider APIs but **not
  run against live GCP** here — flagged in the module docstring and GCP_SETUP.md as validate-on-first-
  deploy. Verified offline: module imports cleanly (lazy SDK imports), `BACKEND=gcp` constructs all
  four providers, and missing config raises a clear `ValueError` (not an auth crash). 4 GCP tests
  added (31 offline tests total).
- **Design notes captured for first deploy:** Document AI online page cap (use batch for 41pp);
  chunk metadata store (Firestore/BigQuery in prod vs in-memory demo map); private endpoint + VPC-SC
  for hardened sovereignty; protobuf skew (§8.1) doesn't occur in the fresh Cloud Run image.
- **Status:** ℹ️ Deploy-ready; unverified against live GCP by design.

### 12.2 ✅ Updated a Phase-2 test for the now-real GCP providers
- `test_providers.py` previously asserted the GCP stubs raise `NotImplementedError`; the Phase-3
  impls raise a clear `ValueError` on missing config instead. Updated the test accordingly.
- **Status:** ✅ Resolved.

---

# ===== PHASE 4A (evaluation harness) =====

## 13. Eval harness

### 13.1 ✅ Built a RAG eval harness (deterministic + LLM-as-judge); all gates pass
- `src/eval/`: golden `dataset.jsonl` (7 Qs incl. 2 negative controls), `judges.py` (deterministic
  refusal/citation-validity/fact checks + a combined LLM-judge returning faithfulness/relevance/
  citation-support in ONE call to save tokens), `harness.py` (runs agent → scores → gates → JSON+MD).
- **Result:** 7/7, all gates pass — behaviour 1.0, citation-validity 1.0, facts 1.0, faithfulness
  1.0, relevance 1.0. New `share_price` negative control also correctly refused.
- **Honest caveat logged:** judge = Gemini grading Gemini (self-judge) → lenient; the *deterministic*
  metrics are the independent evidence. Production: different/stronger judge + human spot-checks +
  adversarial questions. Documented in `docs/LEARNING_LOG.md`.
- 6 eval unit tests (deterministic, no LLM) added to CI. Fixed a `to_markdown` KeyError on missing
  `latency_ms` (made it `.get`).
- **Status:** ✅ Working; caveat documented.

### 13.2 ℹ️ Started docs/LEARNING_LOG.md (user request: follow-along learning md)
- Plain-language walkthrough of each phase (what/why/lesson) + a teaching section on the eval
  metrics and LLM-as-judge. Complements ISSUES_LOG (audit) and REPORT (design).
- **Status:** ℹ️ Ongoing.

---

# ===== PHASE 4B (cost & latency optimization) =====

## 15. Optimization

### 15.1 ✅ Cost instrumentation + semantic answer cache
- Added per-stage **token/cost estimation** (`src/observability/cost.py`) on LLM spans + metrics;
  `/chat` returns `cost_usd`, `/metrics` aggregates tokens/cost. One grounded answer ≈ $0.00013.
- Added a **semantic answer cache** (`src/optimize/semantic_cache.py`, `SEMANTIC_CACHE=on`): embed
  query → cosine ≥ threshold → return cached answer, skipping retrieve+generate.
- **Measured (cross-session):** 8.05s→**0.13s (≈60×)**, cost→**$0**, and it matched a *paraphrase*
  (cosine 0.9935 > 0.97 threshold). Same answer verified.
- **8 optimize unit tests** added. 44 offline tests total, ruff clean.
- **Status:** ✅ Working.

### 15.2 ⚠️→ℹ️ Cache looked broken — I tested the wrong scenario
- First test used ONE agent: after Q1, the agent has history, so the "only cache no-history
  questions" rule (correct — don't cache follow-ups) disabled caching for Q2+. Looked like a bug.
- **Not a bug:** the cache is **cross-session** by design (per-document, process-global). Re-tested
  with two agents/sessions → 60× hit. Documented the single-session limitation in OPTIMIZATION.md.
- **Lesson:** caching an LLM app is a *correctness* problem — define when reuse is safe (same doc, no
  conversational context) before caching.
- **Status:** ℹ️ Clarified + documented.

### 15.3 ℹ️ No-regression argument (didn't re-run full eval)
- Cache returns byte-identical answers; cost tracking is passive. The 7 eval Qs are distinct → all
  cache misses → the exact Phase-4A path (7/7 gates). Skipped re-running eval to spare the Gemini
  free tier; reasoning documented instead. The harness stays the gate for future active optimizations.
- **Status:** ℹ️ Decision recorded.
