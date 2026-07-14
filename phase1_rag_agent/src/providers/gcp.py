"""GCP / Vertex AI provider implementations (Phase 3).

Real, deploy-ready implementations of the provider interfaces against GCP managed services.
Selected with ``BACKEND=gcp`` via the factory; unchanged LangGraph agent runs on top.

    DocumentAIParser      -> Document AI (Layout/OCR)            google.cloud.documentai_v1
    VertexEmbedder        -> Vertex AI Embeddings                vertexai.language_models
    VertexVectorRetriever -> Vertex Vector Search + Ranking API  google.cloud.aiplatform / discoveryengine
    VertexGeminiLLM       -> Gemini on Vertex AI                  vertexai.generative_models

IMPORTANT — honesty note: these were written against the current SDK APIs but have **not** been
run against live GCP services in this environment (no project/credentials here). Treat as
deploy-ready code to validate during the first real deployment. All GCP SDK imports are lazy so the
module imports cleanly without the SDKs installed, and construction is cheap (no network calls).

Auth: uses Application Default Credentials (ADC). Region defaults to asia-south1 (Mumbai) for India
data residency; Document AI uses its own multi-region (us/eu).
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..ingest import Chunk
from ..retriever import RetrievedChunk
from .base import DocumentParser, Embedder, LLMProvider, Retriever


def _require(config: Config, *fields: str) -> None:
    missing = [f for f in fields if not getattr(config, f, "")]
    if missing:
        raise ValueError(
            f"GCP backend needs config: {', '.join(missing)}. Set the matching env vars "
            f"(see GCP_SETUP.md) or run the Terraform in infra/ to create them."
        )


# --------------------------------------------------------------------------- Document AI
class DocumentAIParser(DocumentParser):
    """Document AI processor: layout-aware text + OCR (handles the slide-table problem)."""

    def __init__(self, config: Config):
        self.config = config

    def extract_pages(self, pdf_path: str) -> list[str]:
        _require(self.config, "gcp_project", "docai_processor_id")
        from google.api_core.client_options import ClientOptions
        from google.cloud import documentai_v1 as documentai

        opts = ClientOptions(api_endpoint=f"{self.config.docai_location}-documentai.googleapis.com")
        client = documentai.DocumentProcessorServiceClient(client_options=opts)
        name = client.processor_path(
            self.config.gcp_project, self.config.docai_location, self.config.docai_processor_id
        )
        with open(pdf_path, "rb") as f:
            raw = documentai.RawDocument(content=f.read(), mime_type="application/pdf")
        # NOTE: online process_document has a per-request page cap (~15/30 depending on processor).
        # For larger PDFs use batch_process_documents (GCS in/out); see GCP_SETUP.md.
        result = client.process_document(
            request=documentai.ProcessRequest(name=name, raw_document=raw)
        )
        doc = result.document
        return [self._page_text(page, doc.text) for page in doc.pages]

    @staticmethod
    def _page_text(page, full_text: str) -> str:
        """Reconstruct a page's text from its layout text-anchor segments."""
        segments = getattr(page.layout.text_anchor, "text_segments", [])
        if not segments:
            return ""
        parts = [full_text[int(s.start_index) : int(s.end_index)] for s in segments]
        return "".join(parts).strip()


# --------------------------------------------------------------------------- Embeddings
class VertexEmbedder(Embedder):
    """Vertex AI text embeddings (default text-embedding-005)."""

    _BATCH = 250  # Vertex get_embeddings batch cap

    def __init__(self, config: Config):
        self.config = config
        self._dim = 768
        self._model = None

    def _get_model(self):
        if self._model is None:
            _require(self.config, "gcp_project")
            import vertexai
            from vertexai.language_models import TextEmbeddingModel

            vertexai.init(project=self.config.gcp_project, location=self.config.gcp_location)
            self._model = TextEmbeddingModel.from_pretrained(self.config.vertex_embedding_model)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH):
            batch = texts[i : i + self._BATCH]
            for emb in model.get_embeddings(batch):
                vectors.append(emb.values)
        arr = np.asarray(vectors, dtype="float32")
        # Normalize so inner product == cosine (matches the local FAISS convention).
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)

    @property
    def dim(self) -> int:
        return self._dim


# --------------------------------------------------------------------------- Vector Search + Ranking
class VertexVectorRetriever(Retriever):
    """Vertex AI Vector Search (ANN) + Ranking API (rerank).

    The index/endpoint are provisioned by Terraform (infra/); this class upserts datapoints and
    queries neighbors. Chunk text/metadata is kept in an id->chunk map here for simplicity — a
    production deployment persists that in Firestore/BigQuery (the vector store holds only vectors).
    """

    def __init__(self, config: Config):
        self.config = config
        self._embedder = VertexEmbedder(config)
        self._by_id: dict[str, Chunk] = {}

    def index(self, chunks: list[Chunk]) -> None:
        _require(self.config, "gcp_project", "vector_index_id")
        from google.cloud import aiplatform

        aiplatform.init(project=self.config.gcp_project, location=self.config.gcp_location)
        vectors = self._embedder.embed([c.text for c in chunks])
        datapoints = []
        for c, v in zip(chunks, vectors, strict=True):
            self._by_id[c.chunk_id] = c
            datapoints.append({"datapoint_id": c.chunk_id, "feature_vector": v.tolist()})
        index = aiplatform.MatchingEngineIndex(self.config.vector_index_id)
        index.upsert_datapoints(datapoints=datapoints)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        _require(self.config, "vector_endpoint_id", "vector_deployed_index_id")
        from google.cloud import aiplatform

        qvec = self._embedder.embed([query])[0].tolist()
        endpoint = aiplatform.MatchingEngineIndexEndpoint(self.config.vector_endpoint_id)
        neighbors = endpoint.find_neighbors(
            deployed_index_id=self.config.vector_deployed_index_id,
            queries=[qvec],
            num_neighbors=max(k * 3, k),  # over-fetch, then rerank down to k
        )[0]

        candidates: list[RetrievedChunk] = []
        for n in neighbors:
            chunk = self._by_id.get(n.id)
            if chunk is None:
                continue
            candidates.append(
                RetrievedChunk(
                    chunk=chunk, dense_score=float(n.distance), sparse_score=None, fused_score=0.0
                )
            )
        return self._rerank(query, candidates, k)

    def _rerank(self, query: str, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        try:
            from google.cloud import discoveryengine_v1 as discoveryengine

            client = discoveryengine.RankServiceClient()
            ranking_config = client.ranking_config_path(
                project=self.config.gcp_project,
                location="global",
                ranking_config="default_ranking_config",
            )
            records = [
                discoveryengine.RankingRecord(id=str(i), content=c.chunk.text)
                for i, c in enumerate(candidates)
            ]
            resp = client.rank(
                request=discoveryengine.RankRequest(
                    ranking_config=ranking_config,
                    model=self.config.vertex_ranking_model,
                    query=query,
                    records=records,
                )
            )
            order = {int(r.id): r.score for r in resp.records}
            for i, c in enumerate(candidates):
                c.rerank_score = float(order.get(i, 0.0))
            candidates.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        except Exception:  # noqa: BLE001 — if Ranking API is unavailable, fall back to ANN order
            pass
        return candidates[:k]

    @property
    def num_chunks(self) -> int:
        return len(self._by_id)

    def all_chunks(self) -> list[Chunk]:
        return list(self._by_id.values())


# --------------------------------------------------------------------------- Gemini on Vertex
class VertexGeminiLLM(LLMProvider):
    """Gemini via Vertex AI (sovereign-region capable, enterprise governance)."""

    def __init__(self, config: Config):
        self.config = config
        self._model_cache: dict[str, object] = {}

    def _model(self, system: str):
        if system not in self._model_cache:
            _require(self.config, "gcp_project")
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=self.config.gcp_project, location=self.config.gcp_location)
            self._model_cache[system] = GenerativeModel(
                self.config.vertex_gemini_model, system_instruction=system
            )
        return self._model_cache[system]

    def generate(self, prompt: str, system: str) -> str:
        model = self._model(system)
        resp = model.generate_content(prompt, generation_config={"temperature": 0.0})
        return resp.text.strip()
