"""Provider factory — selects local vs GCP implementations from config.backend.

Phase 3 flips BACKEND=local -> gcp and everything downstream (the LangGraph agent, the API)
keeps working against the same interfaces.
"""

from ..config import Config
from .base import DocumentParser, Embedder, LLMProvider, Retriever


def _backend(config: Config) -> str:
    if config.backend not in ("local", "gcp"):
        raise ValueError(f"Unknown BACKEND '{config.backend}'. Use 'local' or 'gcp'.")
    return config.backend


def build_document_parser(config: Config) -> DocumentParser:
    if _backend(config) == "gcp":
        from .gcp import DocumentAIParser

        return DocumentAIParser(config)
    # Local: default PyMuPDF, but PARSER can select pdfplumber/docling/easyocr/vlm.
    if config.parser == "pymupdf":
        from .local import PyMuPDFParser

        return PyMuPDFParser()
    from .parsers import get_parser

    return get_parser(config.parser)


def build_embedder(config: Config) -> Embedder:
    if _backend(config) == "gcp":
        from .gcp import VertexEmbedder

        return VertexEmbedder(config)
    from .local import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(config)


def build_retriever(config: Config) -> Retriever:
    if _backend(config) == "gcp":
        from .gcp import VertexVectorRetriever

        return VertexVectorRetriever(config)
    from .local import LocalHybridRetriever

    return LocalHybridRetriever(config)


def build_llm(config: Config) -> LLMProvider:
    if _backend(config) == "gcp":
        from .gcp import VertexGeminiLLM

        return VertexGeminiLLM(config)
    from .local import LocalLLM

    return LocalLLM(config)
