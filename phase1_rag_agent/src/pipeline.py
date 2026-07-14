"""Phase 2 assembly: build a ready-to-chat graph agent from a PDF, using the selected backend.

This is the Phase 2 analogue of Phase 1's `get_or_build_index` + `RAGAgent`, but everything
flows through the provider factory so `BACKEND=local|gcp` picks the implementations.
"""

from .config import Config
from .graph import RAGGraphAgent
from .ingest import chunk_pages
from .providers.base import Retriever
from .providers.factory import build_document_parser, build_llm, build_retriever


def index_document(pdf_path: str, config: Config) -> Retriever:
    """Parse + chunk + index a PDF, returning a ready-to-query retriever (shared across sessions)."""
    parser = build_document_parser(config)
    pages = parser.extract_pages(pdf_path)

    chunks = chunk_pages(pages, config.chunk_size_chars, config.chunk_overlap_chars)
    if not chunks:
        raise ValueError(f"No extractable text found in {pdf_path}")

    retriever = build_retriever(config)
    retriever.index(chunks)
    return retriever


def new_agent(retriever: Retriever, config: Config) -> RAGGraphAgent:
    """Create a fresh multi-turn agent (own history) over an already-indexed retriever."""
    return RAGGraphAgent(retriever=retriever, llm=build_llm(config), config=config)


def build_agent(pdf_path: str, config: Config) -> RAGGraphAgent:
    """Convenience: index a PDF and return a single agent (used by the CLI)."""
    return new_agent(index_document(pdf_path, config), config)
