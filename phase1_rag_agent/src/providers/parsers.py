"""Multiple visual-PDF parsing strategies behind the DocumentParser interface.

The earnings deck is a *slide* deck: figures are laid out spatially, so plain reading-order text
(PyMuPDF) can separate a label from its number. This module offers several methods so we can pick
the best per document type, and so the GCP path (Document AI) has a clear local analogue.

Methods (all implement `DocumentParser.extract_pages`):
  PyMuPDFParser     — fast baseline reading-order text (in src/providers/local.py)
  PdfPlumberParser  — pdfplumber word/line extraction (better column handling on real PDFs)
  DoclingParser     — IBM Docling: layout + table-structure recognition -> Markdown tables
  EasyOCRParser     — render page -> OCR (for scanned/image-only pages)
  VLMParser         — render page -> local vision model (moondream via Ollama) transcription

Heavy deps are imported lazily so importing this module stays cheap.

GCP mapping: all of these collapse to **Document AI** (Layout/OCR/Form/Table parsers) in Phase 3.
"""

from __future__ import annotations

import base64
import json
import urllib.request

from ..config import CONFIG
from .base import DocumentParser


def _render_page_png(pdf_path: str, page_index: int, dpi: int = 150) -> bytes:
    import fitz

    doc = fitz.open(pdf_path)
    page = doc[page_index]
    pix = page.get_pixmap(dpi=dpi)
    data = pix.tobytes("png")
    doc.close()
    return data


class PdfPlumberParser(DocumentParser):
    def extract_pages(self, pdf_path: str) -> list[str]:
        import pdfplumber

        pages: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return pages


class DoclingParser(DocumentParser):
    """Layout + table-structure aware. Returns Markdown (tables become pipe tables)."""

    def extract_pages(self, pdf_path: str) -> list[str]:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        # Docling's page split is non-trivial; for parsing-quality comparison we return the whole
        # document's Markdown as a single logical "page". (Per-page citation mapping is a Phase-3
        # Document AI concern.)
        return [result.document.export_to_markdown()]


class EasyOCRParser(DocumentParser):
    """Render each page to an image and OCR it (for scanned/image-only PDFs)."""

    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def extract_pages(self, pdf_path: str) -> list[str]:
        import easyocr
        import fitz

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        doc = fitz.open(pdf_path)
        pages: list[str] = []
        for i in range(doc.page_count):
            png = _render_page_png(pdf_path, i, self.dpi)
            lines = reader.readtext(png, detail=0, paragraph=True)
            pages.append("\n".join(lines))
        doc.close()
        return pages


class VLMParser(DocumentParser):
    """Transcribe each page with a local vision model (moondream via Ollama)."""

    def __init__(self, model: str = "moondream:latest", dpi: int = 150):
        self.model = model
        self.dpi = dpi

    def _transcribe(self, png: bytes) -> str:
        payload = {
            "model": self.model,
            "prompt": (
                "Transcribe ALL text and numbers from this slide exactly, keeping each label "
                "next to its value. Output plain text only."
            ),
            "images": [base64.b64encode(png).decode()],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        req = urllib.request.Request(
            f"{CONFIG.ollama_host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read()).get("response", "").strip()

    def extract_pages(self, pdf_path: str) -> list[str]:
        import fitz

        doc = fitz.open(pdf_path)
        pages: list[str] = []
        for i in range(doc.page_count):
            pages.append(self._transcribe(_render_page_png(pdf_path, i, self.dpi)))
        doc.close()
        return pages


# Registry for the benchmark + factory selection.
def get_parser(name: str) -> DocumentParser:
    name = name.lower()
    if name == "pymupdf":
        from .local import PyMuPDFParser

        return PyMuPDFParser()
    if name == "pdfplumber":
        return PdfPlumberParser()
    if name == "docling":
        return DoclingParser()
    if name == "easyocr":
        return EasyOCRParser()
    if name == "vlm":
        return VLMParser()
    raise ValueError(
        f"Unknown parser '{name}'. Options: pymupdf, pdfplumber, docling, easyocr, vlm"
    )
