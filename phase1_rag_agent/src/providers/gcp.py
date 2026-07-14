"""GCP / Vertex AI provider stubs — the Phase 3 target.

These are intentionally *not wired* yet: they document the exact managed-service and SDK for each
seam and raise a clear, actionable error until Phase 3 provides credentials and fills the body.
The LangGraph agent and the factory already treat these as first-class, so Phase 3 is: implement
the bodies + set `BACKEND=gcp` + provide ADC — no changes to the graph.

Why stubs and not live code here (Phase 2 = "GCP-shaped but local"):
  1. No GCP project/credentials are provisioned in this local environment.
  2. This box's `protobuf` (6.33.5) has a C-extension/pure-Python skew that breaks importing
     `langchain-google-vertexai` / `langchain-google-genai` and some `google-*` proto paths
     (see ISSUES_LOG.md §8). The Phase 3 cloud image will pin a matched protobuf.

Mapping (interface -> managed service -> SDK entry point):
  DocumentAIParser        -> Document AI            google.cloud.documentai_v1.DocumentProcessorServiceClient
  VertexEmbedder          -> Vertex AI Embeddings   vertexai.language_models.TextEmbeddingModel ("gemini-embedding-001")
  VertexVectorRetriever   -> Vertex Vector Search   google.cloud.aiplatform.MatchingEngineIndexEndpoint  (+ Ranking API for rerank)
  VertexGeminiLLM         -> Gemini on Vertex        vertexai.generative_models.GenerativeModel
"""

from ..config import Config
from ..ingest import Chunk
from ..retriever import RetrievedChunk
from .base import DocumentParser, Embedder, LLMProvider, Retriever

_NOT_READY = (
    "GCP backend is a Phase-3 stub. To enable: provision a GCP project + Application Default "
    "Credentials, pin a matched protobuf, implement this method, and set BACKEND=gcp. "
    "See src/providers/gcp.py docstring for the exact SDK entry point."
)


class DocumentAIParser(DocumentParser):
    """Document AI Layout/Form parser — structured tables, OCR, layout-aware page text."""

    def __init__(self, config: Config):
        self.config = config  # expects config.gcp_project, gcp_location, docai_processor_id

    def extract_pages(self, pdf_path: str) -> list[str]:
        # from google.cloud import documentai_v1 as documentai
        # client = documentai.DocumentProcessorServiceClient()
        # name = client.processor_path(project, location, processor_id)
        # result = client.process_document(request={"name": name, "raw_document": ...})
        # return [page.layout.text_anchor ... for page in result.document.pages]
        raise NotImplementedError(_NOT_READY)


class VertexEmbedder(Embedder):
    """Vertex AI text embeddings (gemini-embedding-001)."""

    def __init__(self, config: Config):
        self.config = config
        self._dim = 768  # gemini-embedding output dim (configurable)

    def embed(self, texts: list[str]):
        # import vertexai
        # from vertexai.language_models import TextEmbeddingModel
        # vertexai.init(project=..., location=...)
        # model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
        # return np.array([e.values for e in model.get_embeddings(texts)], dtype="float32")
        raise NotImplementedError(_NOT_READY)

    @property
    def dim(self) -> int:
        return self._dim


class VertexVectorRetriever(Retriever):
    """Vertex AI Vector Search for ANN + Ranking API for reranking (managed hybrid)."""

    def __init__(self, config: Config):
        self.config = config
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        # from google.cloud import aiplatform
        # aiplatform.init(project=..., location=...)
        # index = aiplatform.MatchingEngineIndex(config.vector_index_id)
        # index.upsert_datapoints(datapoints=[...embeddings + chunk metadata...])
        raise NotImplementedError(_NOT_READY)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        # endpoint = aiplatform.MatchingEngineIndexEndpoint(config.vector_endpoint_id)
        # neighbors = endpoint.find_neighbors(queries=[query_vec], num_neighbors=k)
        # then rerank via the Vertex Ranking API (discoveryengine RankService)
        raise NotImplementedError(_NOT_READY)

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)


class VertexGeminiLLM(LLMProvider):
    """Gemini on Vertex AI (sovereign-region capable, enterprise governance)."""

    def __init__(self, config: Config):
        self.config = config

    def generate(self, prompt: str, system: str) -> str:
        # import vertexai
        # from vertexai.generative_models import GenerativeModel
        # vertexai.init(project=..., location=...)
        # model = GenerativeModel(config.gemini_model, system_instruction=system)
        # return model.generate_content(prompt, generation_config={"temperature": 0}).text
        raise NotImplementedError(_NOT_READY)
