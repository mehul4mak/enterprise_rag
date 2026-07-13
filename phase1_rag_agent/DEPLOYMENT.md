# Deployment Guide

How to run the Enterprise RAG service locally, in Docker, and notes for hosting it on a server /
cloud. For the design/reasoning see [REPORT.md](REPORT.md); for the model/cost choices see
[COST_AND_MODEL_RESEARCH.md](COST_AND_MODEL_RESEARCH.md).

---

## Deployment modes at a glance

| Mode | Command | LLM | When |
|------|---------|-----|------|
| CLI (local) | `python main.py --pdf ...` | Ollama or Gemini | dev / quick demo |
| API (local) | `uvicorn src.api:app` | Ollama or Gemini | integrate / test the HTTP surface |
| API (Docker) | `docker compose up --build` | Gemini (default) | hosting / reproducible deploy |

---

## 1. Configuration (env vars)

All config is via env (12-factor). Copy `.env.example` → `.env` and edit. Key vars:

| Var | Default | Meaning |
|-----|---------|---------|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `gemini` \| `openai` \| `anthropic` |
| `GOOGLE_API_KEY` | — | required when `LLM_PROVIDER=gemini` |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` | use `*-latest` aliases (see note below) |
| `OLLAMA_HOST` | `http://localhost:11434` | point at a remote GPU box to offload local inference |
| `OLLAMA_MODEL` | `gemma2:2b` | any pulled Ollama model |
| `INDEX_PDF` | — | if set, the API auto-indexes this PDF on startup |

> **Gemini model note:** newly-created API keys can only call the newest (3.x) models; pinned 2.x
> ids return `404 "no longer available to new users"`. Always prefer the version-agnostic
> `gemini-flash-lite-latest` / `gemini-flash-latest` aliases. (See ISSUES_LOG.md §6.6.)

> **Data-privacy note:** the free Gemini tier may use submitted content for product improvement.
> Only send **non-confidential** documents through a free-tier key (the Adani earnings deck used
> here is public investor material). For private data, use a paid-tier key or the local/Ollama mode.

---

## 2. Run the HTTP API

```bash
pip install -r requirements.txt
export LLM_PROVIDER=gemini GOOGLE_API_KEY=...           # or LLM_PROVIDER=ollama
export INDEX_PDF=./data/earnings_presentation_q2fy26.pdf
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path | Body | Purpose |
|--------|------|------|---------|
| GET | `/health` | — | liveness + which provider/index is active |
| POST | `/ingest` | `{"pdf_path": "..."}` | (re)build/load the active index |
| POST | `/chat` | `{"question": "...", "session_id": "?"}` | grounded answer + citations + retrieved debug |
| GET | `/sessions/{id}` | — | conversation history |
| DELETE | `/sessions/{id}` | — | reset a session |

Interactive docs are auto-served at **`/docs`** (Swagger) and **`/redoc`**.

### Example

```bash
# Ask a question (session auto-created; reuse session_id for follow-ups)
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"question":"What is the consolidated total income in H1-26?"}' | jq
```
```json
{
  "session_id": "a1b2...",
  "answer": "The consolidated total income in H1-26 is 44,281 ₹ crore [p22:c35].",
  "citations": ["[p22:c35]"],
  "retrieved": [{"citation": "[p22:c35]", "score": 4.58, "snippet": "..."}]
}
```

---

## 3. Run in Docker (recommended for hosting)

```bash
# .env must contain GOOGLE_API_KEY (compose reads it)
docker compose up --build
# API on http://localhost:8000  (auto-indexes data/earnings_presentation_q2fy26.pdf)
```

The image pre-caches the embedding + reranker models at build time, so containers start fast and
run offline for retrieval. The `.cache` volume persists the built FAISS/BM25 index across restarts.

### Fully offline (no API key) deployment
Uncomment the `ollama` service in `docker-compose.yml`, then:
```bash
export LLM_PROVIDER=ollama OLLAMA_HOST=http://ollama:11434
docker compose up --build
docker compose exec ollama ollama pull gemma2:2b
```

---

## 4. CI/CD

GitHub Actions (repo root `.github/workflows/`):

- **`ci.yml`** — on every push/PR: `ruff check`, `ruff format --check`, and the offline unit tests
  (`test_chunking`, `test_grounding`, `test_api`). The acceptance suite is excluded from CI because
  it needs a live LLM backend.
- **`docker.yml`** — builds the container on pushes to `main`/`phase-1-local-rag` and on `v*` tags;
  pushes to GHCR (`ghcr.io/<owner>/<repo>`) on tags using the built-in `GITHUB_TOKEN`.

---

## 5. Hosting on a server / cloud (generic)

Any host that runs a container works (a small VM, Cloud Run, ECS, a GPU box):

1. Provide `GOOGLE_API_KEY` as a secret env var (never bake it into the image).
2. Mount or bake in the PDF(s); set `INDEX_PDF` (or call `/ingest` after boot).
3. Expose port 8000 behind your load balancer/ingress; `/health` is the readiness probe.
4. Put the index `.cache` on a persistent volume so restarts don't re-embed.

> Phase 2 replaces this generic hosting with **GCP-native** managed services (Cloud Run + Vertex AI
> Vector Search + Gemini on Vertex + Document AI). See [REPORT.md](REPORT.md) §6.
