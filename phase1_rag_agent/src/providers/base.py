"""Provider interfaces — the seams that make local↔GCP a swap, not a rewrite.

Each abstract base maps 1:1 to a GCP-managed service (see src/providers/gcp.py):

    DocumentParser   ->  Document AI (Layout/Form parser)
    Embedder         ->  Vertex AI text-embedding (gemini-embedding)
    Retriever        ->  Vertex AI Vector Search + Ranking API
    LLMProvider      ->  Gemini on Vertex AI

The LangGraph agent (src/graph.py) depends ONLY on these interfaces, so Phase 3 selects
the GCP implementations via the factory and the graph is unchanged.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..ingest import Chunk
from ..retriever import RetrievedChunk


class DocumentParser(ABC):
    """PDF -> per-page text (best effort)."""

    @abstractmethod
    def extract_pages(self, pdf_path: str) -> list[str]:
        """Return page texts, index 0 = page 1."""


class Embedder(ABC):
    """Text -> dense vectors (L2-normalized so inner product == cosine)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...


class Retriever(ABC):
    """Owns indexing + hybrid retrieval + reranking for one document."""

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Build/persist whatever indices this retriever needs."""

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks with scores for a query."""

    @property
    @abstractmethod
    def num_chunks(self) -> int: ...

    @abstractmethod
    def all_chunks(self) -> list[Chunk]: ...


class LLMProvider(ABC):
    """Grounded text generation."""

    @abstractmethod
    def generate(self, prompt: str, system: str) -> str: ...
