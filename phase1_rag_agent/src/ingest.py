"""PDF ingestion: per-page text extraction + page-aware chunking."""

import re
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class Chunk:
    chunk_id: str  # global id, e.g. "c42"
    page: int  # 1-indexed page number
    text: str

    @property
    def citation(self) -> str:
        return f"[p{self.page}:{self.chunk_id}]"


def extract_pages(pdf_path: str) -> list[str]:
    """Best-effort per-page text extraction. Returns list indexed by page (0-based)."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        pages.append(text)
    doc.close()
    return pages


def _pack_lines(lines: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack lines into char-budgeted windows with trailing overlap."""
    chunks: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if current and current_len + len(line) + 1 > chunk_size:
            chunks.append(current)
            overlap_lines: list[str] = []
            overlap_len = 0
            for prev_line in reversed(current):
                if overlap_len + len(prev_line) > overlap:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_len += len(prev_line) + 1
            current = overlap_lines
            current_len = overlap_len
        current.append(line)
        current_len += len(line) + 1

    if current:
        chunks.append(current)

    return ["\n".join(c) for c in chunks]


def chunk_pages(pages: list[str], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk each page's text independently so every chunk carries exactly one page number."""
    chunks: list[Chunk] = []
    counter = 0
    for page_idx, page_text in enumerate(pages):
        page_num = page_idx + 1
        if not page_text.strip():
            continue
        lines = [l for l in page_text.split("\n") if l.strip()]
        for piece in _pack_lines(lines, chunk_size, overlap):
            counter += 1
            chunks.append(Chunk(chunk_id=f"c{counter}", page=page_num, text=piece))
    return chunks


def ingest_pdf(pdf_path: str, chunk_size: int, overlap: int) -> list[Chunk]:
    pages = extract_pages(pdf_path)
    return chunk_pages(pages, chunk_size, overlap)
