"""Local provider implementations — wrap the proven Phase 1 components behind the interfaces.

These run fully on this machine (no cloud). Phase 3 swaps them for the GCP ones in gcp.py
without touching the LangGraph agent.
"""

import numpy as np

from ..config import Config
from ..embeddings import embed_texts, get_embedder
from ..index_store import HybridIndex, get_or_build_from_chunks
from ..ingest import Chunk, extract_pages
from ..llm import generate
from ..retriever import RetrievedChunk, retrieve
from .base import DocumentParser, Embedder, LLMProvider, Retriever


class PyMuPDFParser(DocumentParser):
    """Local PDF text extraction (Phase-3 counterpart: Document AI)."""

    def extract_pages(self, pdf_path: str) -> list[str]:
        return extract_pages(pdf_path)


class SentenceTransformerEmbedder(Embedder):
    """Local embeddings via sentence-transformers (Phase-3 counterpart: Vertex text-embedding)."""

    def __init__(self, config: Config):
        self._model_name = config.embedding_model
        self._dim: int | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        return embed_texts(texts, self._model_name)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = get_embedder(self._model_name).get_sentence_embedding_dimension()
        return self._dim


class LocalHybridRetriever(Retriever):
    """FAISS (dense) + BM25 (sparse) + cross-encoder rerank.

    Phase-3 counterpart: Vertex AI Vector Search + Ranking API.
    """

    def __init__(self, config: Config):
        self.config = config
        self._index: HybridIndex | None = None

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks to index.")
        # Disk-cached: re-running on the same document skips re-embedding.
        self._index = get_or_build_from_chunks(chunks, self.config)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._index is None:
            raise RuntimeError("Retriever.index() must be called before retrieve().")
        return retrieve(query, self._index, self.config, top_k=k)

    @property
    def num_chunks(self) -> int:
        return 0 if self._index is None else len(self._index.chunks)

    def all_chunks(self) -> list[Chunk]:
        return [] if self._index is None else list(self._index.chunks)


class LocalLLM(LLMProvider):
    """Dispatches to Ollama/Gemini/OpenAI/Anthropic via the Phase 1 llm.generate seam.

    Phase-3 counterpart: Gemini on Vertex AI (VertexGeminiLLM in gcp.py).
    """

    def __init__(self, config: Config):
        self.config = config

    def generate(self, prompt: str, system: str) -> str:
        return generate(prompt, system, self.config)
