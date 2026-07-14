# Visual-PDF Parsing — Methods & Comparison

The assessment deck is a **slide deck**: numbers are laid out spatially, so plain reading-order text
can separate a label ("Total Income") from its value (49,263 / 44,281). This is the single biggest
risk to grounded numeric QA, so Phase 2 tries **five** parsing strategies behind the same
`DocumentParser` interface (`src/providers/parsers.py`) and compares them.

Run the comparison:
```bash
pip install -r requirements-parsers.txt
python -m tests.parser_benchmark --methods pymupdf,pdfplumber,docling,easyocr,vlm
```

## Methods

| Method | How | Best for | Cost |
|--------|-----|----------|------|
| **pymupdf** | reading-order text (baseline) | fast, digital-native PDFs | ~0.1s/doc |
| **pdfplumber** | word/line geometry | column/table layout on real PDFs | ~1s/doc |
| **docling** | layout + table-structure model → Markdown tables | tables, structure | model load + ~s/page |
| **easyocr** | render page → OCR | scanned / image-only pages | ~s/page (CPU) |
| **vlm** | render page → moondream (Ollama) transcription | pictorial slides, charts | slow (VLM/page) |

Each collapses to **Document AI** (Layout/OCR/Form/Table parsers) in the Phase 3 GCP path.

## Scoring

Two association checks on the extracted text (what matters for numeric grounding):
- **label_near_number** — is "Total Income" within ~80 chars of `44,281` / `49,263`?
- **has_44281 / has_49263** — are the H1-26 / H1-25 total-income figures present at all?

## Results

Measured on this machine (12-core CPU, no GPU) against `data/earnings_presentation_q2fy26.pdf`.
The association check targets the page-2 "Total Income" figures (H1-25 49,263 / H1-26 44,281).

| Method | Scope | Time | label↔number adjacent | has 44,281 | has 49,263 |
|--------|-------|------|:---------------------:|:----------:|:----------:|
| pymupdf | full doc (41pp) | 0.5 s | ❌ | ✅ | ✅ |
| **pdfplumber** | full doc (41pp) | 13 s | ✅ | ✅ | ✅ |
| docling | full doc | 223 s | ❌ | ❌ | ❌ |
| easyocr | page 2 only | ~37 s (incl. model dl) | ✅ | ✅ | ✅ |
| vlm (moondream) | page 2 only | 69 s | ❌ | ❌ | ❌ |

Notes:
- **pymupdf** has the numbers but *not adjacent to their label* — this is exactly the Phase 1
  limitation. It still passed acceptance because chunking keeps label+number in the same chunk.
- **pdfplumber** is the only *full-document* method that keeps the label next to its value, at a
  modest 13 s. Best cost/quality here.
- **docling** was the surprise: 223 s and it **lost** the key figures entirely. Its layout/table
  model expects real tables; on these *infographic* slides (numbers as positioned/vector text) it
  under-performed simple text extraction. A good reminder that "fancier" ≠ "better" per document type.
- **easyocr** (render→OCR) recovers the numbers *and* their adjacency because it reads the page
  visually — but it's per-page slow and overkill for a digital-native PDF. It's the right tool for
  **scanned/image-only** documents.
- **vlm (moondream)** captured the slide's *labels* ("ADANI … PBT EBITDA TOTAL INCOME …") but **not
  the numbers** (69 s/page). A small 1.7B vision model isn't enough for dense financial figures — a
  larger VLM (Gemini vision, or a 7B+ local VLM) would be needed. Implemented as a method to try;
  not recommended for numeric decks at this model size.

## Recommendation

- **Default:** keep `PARSER=pymupdf` for speed; chunking already preserves label+number locality
  well enough that acceptance passes.
- **For better numeric locality / per-sentence citations:** `PARSER=pdfplumber` (13 s, best
  full-doc association).
- **For scanned/image PDFs:** `PARSER=easyocr` (or `vlm` for chart-heavy slides).
- **Phase 3:** all of these converge on **Document AI**, whose table/layout parsers are purpose-built
  for exactly the infographic-table problem docling stumbled on here.

Select at runtime: `PARSER=pdfplumber python main.py --pdf ...` (needs `requirements-parsers.txt`).
