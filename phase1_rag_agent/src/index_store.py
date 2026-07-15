"""Hybrid (dense + sparse) retrieval index: build once, cache on disk, reload instantly.

The cache is keyed on exactly what determines the index — the chunk texts + chunking params +
embedding model — so any change to parsing, chunking, or model invalidates it automatically.
(Phase 1 keyed the cache on the PDF file bytes; keying on chunks is stricter and lets every
DocumentParser implementation share the same cache machinery.)
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi

from .config import CACHE_DIR, Config
from .embeddings import embed_texts
from .ingest import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class HybridIndex:
    chunks: list[Chunk]
    faiss_index: faiss.Index
    bm25: BM25Okapi
    source_pdf: str


def _cache_key(chunks: list[Chunk], config: Config) -> str:
    h = hashlib.md5()
    for c in chunks:
        h.update(f"{c.page}:{c.chunk_id}:".encode())
        h.update(c.text.encode())
    h.update(config.embedding_model.encode())
    return h.hexdigest()[:16]


def build_from_chunks(chunks: list[Chunk], config: Config, source_pdf: str = "") -> HybridIndex:
    """Build an in-memory hybrid index directly from chunks."""
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


def get_or_build_from_chunks(
    chunks: list[Chunk], config: Config, source_pdf: str = ""
) -> HybridIndex:
    """Disk-cached index build: instant reload when chunks + embedding model are unchanged."""
    cache_dir = CACHE_DIR / _cache_key(chunks, config)
    cached = _load(cache_dir)
    if cached is not None:
        return cached
    index = build_from_chunks(chunks, config, source_pdf=source_pdf)
    _save(index, cache_dir)
    return index
