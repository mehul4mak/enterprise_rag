# Phase 3 — Deploy to GCP (asia-south1 / Mumbai)

Deploys the same agent with `BACKEND=gcp`: Document AI parsing, Vertex embeddings, Vertex Vector
Search + Ranking, and Gemini on Vertex — in an India-sovereign region.

> **Status / honesty:** the provider code (`src/providers/gcp.py`) and this IaC are written against
> the current SDK/Terraform provider APIs but were **not run against live GCP** in development (no
> project/credentials there). Validate on first deploy. Everything is structured so the app,
> LangGraph agent, and governance layer are unchanged from Phase 2 — only the backend swaps.

> **Cost warning:** a deployed Vector Search index endpoint bills **hourly** while it exists
> (dedicated replicas), plus per-call Vertex/Document AI usage. Run `terraform destroy` when done.

---

## 0. Prerequisites
- A GCP project with **billing enabled**.
- `gcloud`, `terraform` (>=1.5), and Docker installed locally.
- Authenticate:
  ```bash
  gcloud auth login
  gcloud auth application-default login
  gcloud config set project YOUR_PROJECT_ID
  ```

## 1. Provision infrastructure (Terraform)
```bash
cd infra/terraform
terraform init
terraform apply -var project_id=YOUR_PROJECT_ID -var region=asia-south1
```
This creates: enabled APIs, a GCS bucket, a **Document AI** layout processor, a **Vertex Vector
Search** index + endpoint + deployment, an Artifact Registry repo, a **KMS** key (Phase-4 PII
vault), a runtime service account with least-privilege IAM, and the Cloud Run service.

Capture the outputs (they are the app's env vars):
```bash
terraform output   # docai_processor_id, vector_index_id, vector_endpoint_id, vector_deployed_index_id, ...
```

## 2. Build & deploy the container
```bash
cd ../..                     # repo root of the app
gcloud builds submit --config cloudbuild.yaml --substitutions _REGION=asia-south1
```
`cloudbuild.yaml` builds `Dockerfile.gcp`, pushes to Artifact Registry, and deploys to Cloud Run.
(Terraform already set the `BACKEND=gcp` + resource-id env vars on the service; Cloud Build just
rolls the new image.)

## 3. Smoke test
```bash
URL=$(gcloud run services describe enterprise-rag --region asia-south1 --format 'value(status.url)')
curl -s $URL/health | jq
curl -s $URL/ingest -H 'Content-Type: application/json' \
  -d '{"pdf_path":"gs://YOUR_BUCKET/earnings.pdf"}' | jq   # or a container-local path
curl -s $URL/chat   -H 'Content-Type: application/json' \
  -d '{"question":"What is the consolidated total income in H1-26?"}' | jq
```

---

## Config → resource mapping

| Env var | Source | Used by |
|---------|--------|---------|
| `GCP_PROJECT` / `GCP_LOCATION` | you / `asia-south1` | all Vertex calls |
| `DOCAI_LOCATION` / `DOCAI_PROCESSOR_ID` | `us` / TF output | Document AI |
| `VERTEX_EMBEDDING_MODEL` | `text-embedding-005` | embeddings |
| `VERTEX_RANKING_MODEL` | `semantic-ranker-default@latest` | rerank |
| `VERTEX_GEMINI_MODEL` | `gemini-2.0-flash-001` | generation |
| `VECTOR_INDEX_ID` / `VECTOR_ENDPOINT_ID` / `VECTOR_DEPLOYED_INDEX_ID` | TF outputs | Vector Search |

## Known deployment considerations (validate on first run)
- **Document AI page cap:** online `process_document` caps pages (~15–30). For the 41-page deck use
  `batch_process_documents` (GCS in/out) — noted in `gcp.py`. The Layout parser is best for the
  slide-table problem (it's the managed fix for what docling missed locally, see PARSING.md).
- **Chunk metadata store:** `VertexVectorRetriever` keeps an in-memory `id→chunk` map for the demo.
  Production should persist chunk text/metadata in **Firestore/BigQuery** (Vector Search stores only
  vectors) so retrieval survives restarts and scales across instances.
- **Sovereignty hardening:** switch the index endpoint to **PRIVATE** (VPC + Private Service
  Connect) and add a **VPC-SC** perimeter + Org Policy `resource-locations` = India. `DATA_RESIDENCY`
  is set to `regional` on the service; set `strict` only with an in-region/local model.
- **protobuf:** the local dev skew (ISSUES_LOG §8.1) does not occur in the fresh Cloud Run image.

## Teardown
```bash
cd infra/terraform && terraform destroy -var project_id=YOUR_PROJECT_ID
```
