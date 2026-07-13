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

## 6. LLM / generation issues

### 6.1 ✅ Local Ollama path verified
- Smoke test: `generate("Reply with exactly the word: OK", ...)` via `gemma2:2b` returned `'OK'`. Local, offline, zero-cost generation path confirmed working before wiring the RAG prompt.
- **Status:** ✅ Working.

### 6.2 ⚠️→✅ Default Gemini model was already deprecated
- **Caught during cost research:** I had defaulted `GEMINI_MODEL=gemini-2.0-flash`. Web check (July 2026) showed **Gemini 2.0 Flash / Flash-Lite were shut down 2026-06-01**. Had we shipped that default, the Gemini path would 404 at runtime.
- **Fix:** Changed default to `gemini-2.5-flash-lite` (cheapest active model, $0.10/$0.40 per 1M) in both `config.py` and `.env.example`, with an inline comment noting the deprecation.
- **Lesson:** Never hardcode a model id from memory — verify it's live. (This is exactly why the repo pins model ids in one config file, not scattered in code.)
- **Status:** ✅ Resolved.

### 6.3 ℹ️ Cost sizing → API is effectively free for this workload
- Sized the acceptance suite at ~1,500 input + 250 output tokens/question. 500 dev calls ≈ **$0.13**. Documented full analysis + GPU-rental alternative in `COST_AND_MODEL_RESEARCH.md`.
- **Decision:** default stays local `gemma2:2b` ($0, offline proof); recommend Gemini 2.5 Flash-Lite for the quality/eval tier (aligns with Phase-2 GCP/Gemini).
- **Status:** ℹ️ Decision recorded.

---

## 7. Result-quality / acceptance-test issues

### 7.1 ✅ Numeric question passed on the local 2B model (better than expected)
- **Q:** "What is the consolidated total income in H1-26?"
- **A:** `44,281 ₹ crore [p22:c35]` — correct value, correct period (H1-26 not Q2), valid citation to a retrieved chunk.
- **Why notable:** I expected a 2B model to fumble numeric extraction from noisy slide text. It didn't, because retrieval+rerank handed it the *right* chunk and the strict prompt forced verbatim copying. This validates the "retrieval does the heavy lifting, LLM just extracts+cites" design — small model is viable for the pipeline proof.
- **Status:** ✅ Pass.
