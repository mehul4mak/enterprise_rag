import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"


@dataclass(frozen=True)
class Config:
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()

    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma2:2b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size_chars: int = 1000
    chunk_overlap_chars: int = 150

    top_k_dense: int = 15
    top_k_sparse: int = 15
    top_k_final: int = 5

    max_history_turns: int = 6

    # --- Phase 2: backend selection (local | gcp) ---
    backend: str = os.getenv("BACKEND", "local").lower()
    # local PDF parser: pymupdf (fast default) | pdfplumber | docling | easyocr | vlm
    parser: str = os.getenv("PARSER", "pymupdf").lower()

    # --- Governance / safety toggles ---
    guardrails_enabled: bool = os.getenv("GUARDRAILS", "on").lower() in ("on", "1", "true")
    model_armor_enabled: bool = os.getenv("MODEL_ARMOR", "on").lower() in ("on", "1", "true")
    # data residency: off | regional | strict  (strict = local-only, blocks external LLM egress)
    data_residency: str = os.getenv("DATA_RESIDENCY", "off").lower()

    # --- Phase 4B: optimization ---
    semantic_cache: bool = os.getenv("SEMANTIC_CACHE", "off").lower() in ("on", "1", "true")
    cache_threshold: float = float(os.getenv("CACHE_THRESHOLD", "0.97"))

    # --- Phase 3 GCP settings (consumed by src/providers/gcp.py) ---
    gcp_project: str = os.getenv("GCP_PROJECT", "")
    gcp_location: str = os.getenv("GCP_LOCATION", "asia-south1")  # Mumbai (India-sovereign)
    docai_location: str = os.getenv("DOCAI_LOCATION", "us")  # Document AI multi-region: us | eu
    docai_processor_id: str = os.getenv("DOCAI_PROCESSOR_ID", "")
    vertex_embedding_model: str = os.getenv("VERTEX_EMBEDDING_MODEL", "text-embedding-005")
    vertex_ranking_model: str = os.getenv("VERTEX_RANKING_MODEL", "semantic-ranker-default@latest")
    vertex_gemini_model: str = os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.0-flash-001")
    vector_index_id: str = os.getenv("VECTOR_INDEX_ID", "")
    vector_endpoint_id: str = os.getenv("VECTOR_ENDPOINT_ID", "")
    vector_deployed_index_id: str = os.getenv("VECTOR_DEPLOYED_INDEX_ID", "")

    @property
    def active_model(self) -> str:
        """The model string used for the current provider (for cost attribution)."""
        return {
            "gemini": self.gemini_model,
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
            "ollama": self.ollama_model,
        }.get(self.llm_provider, self.llm_provider)


CONFIG = Config()
