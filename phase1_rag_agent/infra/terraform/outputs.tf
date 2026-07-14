# These map directly to the app's env vars (see GCP_SETUP.md).

output "docai_processor_id" {
  value       = google_document_ai_processor.layout.processor_id
  description = "DOCAI_PROCESSOR_ID"
}

output "vector_index_id" {
  value       = google_vertex_ai_index.rag.id
  description = "VECTOR_INDEX_ID"
}

output "vector_endpoint_id" {
  value       = google_vertex_ai_index_endpoint.rag.id
  description = "VECTOR_ENDPOINT_ID"
}

output "vector_deployed_index_id" {
  value       = "rag_deployed"
  description = "VECTOR_DEPLOYED_INDEX_ID"
}

output "artifact_registry" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
  description = "Docker image prefix."
}

output "cloud_run_url" {
  value       = google_cloud_run_v2_service.api.uri
  description = "Deployed service URL."
}

output "runtime_service_account" {
  value       = google_service_account.runtime.email
  description = "Cloud Run runtime SA."
}

output "pii_vault_kms_key" {
  value       = google_kms_crypto_key.pii_vault.id
  description = "KMS key for the Phase 4 PII vault."
}
