############################################################
# Enterprise RAG — Phase 3 infrastructure (asia-south1)
# Provisions: APIs, GCS, Document AI, Vertex Vector Search
# (index + endpoint + deployment), Artifact Registry, KMS
# (PII vault), a runtime service account, and Cloud Run.
############################################################

locals {
  services = [
    "aiplatform.googleapis.com",
    "documentai.googleapis.com",
    "discoveryengine.googleapis.com",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudkms.googleapis.com",
    "storage.googleapis.com",
  ]
}

resource "google_project_service" "enabled" {
  for_each                   = toset(local.services)
  service                    = each.value
  disable_dependent_services = false
  disable_on_destroy         = false
}

# ------------------------------------------------------------------ GCS
# Holds the Vector Search index seed data and (optionally) source PDFs.
resource "google_storage_bucket" "index_data" {
  name                        = "${var.project_id}-rag-index"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true
  depends_on                  = [google_project_service.enabled]
}

# Empty seed so the index has a contents_delta_uri to point at initially.
resource "google_storage_bucket_object" "index_seed" {
  bucket  = google_storage_bucket.index_data.name
  name    = "index/seed/.keep"
  content = "seed"
}

# ------------------------------------------------------------------ Document AI
resource "google_document_ai_processor" "layout" {
  location     = var.docai_location
  display_name = "rag-layout-parser"
  type         = "LAYOUT_PARSER_PROCESSOR" # layout + tables; use OCR_PROCESSOR for scans
  depends_on   = [google_project_service.enabled]
}

# ------------------------------------------------------------------ Vertex Vector Search
resource "google_vertex_ai_index" "rag" {
  region       = var.region
  display_name = "rag-index"
  description  = "Enterprise RAG chunk embeddings"
  index_update_method = "STREAM_UPDATE" # allow upsert_datapoints from the app

  metadata {
    contents_delta_uri = "gs://${google_storage_bucket.index_data.name}/index/seed"
    config {
      dimensions                  = var.embedding_dim
      approximate_neighbors_count = 150
      distance_measure_type       = "DOT_PRODUCT_DISTANCE" # matches normalized (cosine) embeddings
      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count    = 500
          leaf_nodes_to_search_percent = 10
        }
      }
    }
  }
  depends_on = [google_storage_bucket_object.index_seed]
}

resource "google_vertex_ai_index_endpoint" "rag" {
  region       = var.region
  display_name = "rag-endpoint"
  # Public endpoint for demo simplicity. For sovereignty/production, use a PRIVATE endpoint
  # inside a VPC + Private Service Connect (set network / private_service_connect_config).
  public_endpoint_enabled = true
  depends_on              = [google_project_service.enabled]
}

resource "google_vertex_ai_index_endpoint_deployed_index" "rag" {
  index_endpoint    = google_vertex_ai_index_endpoint.rag.id
  index             = google_vertex_ai_index.rag.id
  deployed_index_id = "rag_deployed"
  display_name      = "rag-deployed"

  dedicated_resources {
    machine_spec {
      machine_type = "e2-standard-2"
    }
    min_replica_count = 1
    max_replica_count = 2
  }
}

# ------------------------------------------------------------------ Artifact Registry
resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = "rag"
  format        = "DOCKER"
  depends_on    = [google_project_service.enabled]
}

# ------------------------------------------------------------------ KMS (PII vault — Phase 4 groundwork)
resource "google_kms_key_ring" "rag" {
  name       = "rag-keyring"
  location   = var.region
  depends_on = [google_project_service.enabled]
}

resource "google_kms_crypto_key" "pii_vault" {
  name            = "pii-vault"
  key_ring        = google_kms_key_ring.rag.id
  rotation_period = "7776000s" # 90 days
}

# ------------------------------------------------------------------ Runtime service account + IAM
resource "google_service_account" "runtime" {
  account_id   = "rag-runtime"
  display_name = "Enterprise RAG Cloud Run runtime"
}

resource "google_project_iam_member" "runtime_roles" {
  for_each = toset([
    "roles/aiplatform.user",       # Vertex embeddings, Vector Search, Gemini
    "roles/documentai.apiUser",    # Document AI
    "roles/discoveryengine.viewer",# Ranking API
    "roles/storage.objectViewer",  # read index data / docs
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

# ------------------------------------------------------------------ Cloud Run
resource "google_cloud_run_v2_service" "api" {
  name     = "enterprise-rag"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.runtime.email
    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }
    containers {
      image = var.image != "" ? var.image : "${var.region}-docker.pkg.dev/${var.project_id}/rag/enterprise-rag:latest"
      ports {
        container_port = 8000
      }
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
      env {
        name  = "BACKEND"
        value = "gcp"
      }
      env {
        name  = "LLM_PROVIDER"
        value = "gemini"
      }
      env {
        name  = "GCP_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GCP_LOCATION"
        value = var.region
      }
      env {
        name  = "DOCAI_LOCATION"
        value = var.docai_location
      }
      env {
        name  = "DOCAI_PROCESSOR_ID"
        value = google_document_ai_processor.layout.processor_id
      }
      env {
        name  = "VECTOR_INDEX_ID"
        value = google_vertex_ai_index.rag.id
      }
      env {
        name  = "VECTOR_ENDPOINT_ID"
        value = google_vertex_ai_index_endpoint.rag.id
      }
      env {
        name  = "VECTOR_DEPLOYED_INDEX_ID"
        value = "rag_deployed"
      }
      env {
        name  = "DATA_RESIDENCY"
        value = "regional"
      }
    }
  }
  depends_on = [google_project_service.enabled]
}
