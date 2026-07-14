variable "project_id" {
  type        = string
  description = "GCP project id."
}

variable "region" {
  type        = string
  default     = "asia-south1" # Mumbai — India data residency (sovereign)
  description = "Region for Vertex AI, Vector Search, Cloud Run, KMS, GCS."
}

variable "docai_location" {
  type        = string
  default     = "us" # Document AI multi-region: us | eu
  description = "Document AI processor location (own multi-region)."
}

variable "embedding_dim" {
  type        = number
  default     = 768 # text-embedding-005
  description = "Vector Search index dimensionality; must match VERTEX_EMBEDDING_MODEL output."
}

variable "image" {
  type        = string
  description = "Container image (Artifact Registry) for the Cloud Run service."
  default     = ""
}
