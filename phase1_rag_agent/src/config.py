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

    # --- Phase 3 GCP settings (stubs; consumed by src/providers/gcp.py) ---
    gcp_project: str = os.getenv("GCP_PROJECT", "")
    gcp_location: str = os.getenv("GCP_LOCATION", "us-central1")
    docai_processor_id: str = os.getenv("DOCAI_PROCESSOR_ID", "")
    vector_index_id: str = os.getenv("VECTOR_INDEX_ID", "")
    vector_endpoint_id: str = os.getenv("VECTOR_ENDPOINT_ID", "")


CONFIG = Config()
