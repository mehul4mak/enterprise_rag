"""Hybrid (dense + sparse) retrieval index: build, persist, and reload per-PDF."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi

from .config import CACHE_DIR, Config
from .embeddings import embed_texts
from .ingest import Chunk, ingest_pdf

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class HybridIndex:
    chunks: list[Chunk]
    faiss_index: faiss.Index
    bm25: BM25Okapi
    source_pdf: str


def _cache_key(pdf_path: str, config: Config) -> str:
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        h.update(f.read())
    h.update(str(config.chunk_size_chars).encode())
    h.update(str(config.chunk_overlap_chars).encode())
    h.update(config.embedding_model.encode())
    return h.hexdigest()[:16]


def _build(chunks: list[Chunk], config: Config, source_pdf: str) -> HybridIndex:
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts, config.embedding_model)
    dim = vectors.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(vectors)
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    return HybridIndex(chunks=chunks, faiss_index=faiss_index, bm25=bm25, source_pdf=source_pdf)


def _save(index: HybridIndex, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index.faiss_index, str(cache_dir / "index.faiss"))
    meta = {
        "source_pdf": index.source_pdf,
        "chunks": [{"chunk_id": c.chunk_id, "page": c.page, "text": c.text} for c in index.chunks],
    }
    (cache_dir / "chunks.json").write_text(json.dumps(meta))


def _load(cache_dir: Path) -> HybridIndex | None:
    index_file = cache_dir / "index.faiss"
    meta_file = cache_dir / "chunks.json"
    if not (index_file.exists() and meta_file.exists()):
        return None
    meta = json.loads(meta_file.read_text())
    chunks = [Chunk(chunk_id=c["chunk_id"], page=c["page"], text=c["text"]) for c in meta["chunks"]]
    faiss_index = faiss.read_index(str(index_file))
    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
    return HybridIndex(
        chunks=chunks, faiss_index=faiss_index, bm25=bm25, source_pdf=meta["source_pdf"]
    )


def get_or_build_index(pdf_path: str, config: Config, force_reindex: bool = False) -> HybridIndex:
    key = _cache_key(pdf_path, config)
    cache_dir = CACHE_DIR / key

    if not force_reindex:
        cached = _load(cache_dir)
        if cached is not None:
            return cached

    chunks = ingest_pdf(pdf_path, config.chunk_size_chars, config.chunk_overlap_chars)
    if not chunks:
        raise ValueError(f"No extractable text found in {pdf_path}")
    index = _build(chunks, config, source_pdf=pdf_path)
    _save(index, cache_dir)
    return index
