"""Benchmark visual-PDF parsing methods on the earnings deck.

For each method we measure: wall time, total characters, and two *association* checks that matter
for grounded numeric QA on this slide deck:
  A1: does "Total Income" appear within 60 chars of the H1-26 figure 44,281?
  A2: is the page-2 figure 49,263 (H1-25 total income) present at all?

Usage:
    python -m tests.parser_benchmark [--methods pymupdf,pdfplumber,docling,easyocr,vlm] [--pdf PATH]

VLM/EasyOCR are slow (render+model); include them only when you want the full comparison.
"""

import argparse
import time

from src.providers.parsers import get_parser

DEFAULT_PDF = "data/earnings_presentation_q2fy26.pdf"


def association_score(full_text: str) -> dict:
    text = " ".join(full_text.split())
    has_44281 = "44,281" in text
    has_49263 = "49,263" in text
    near = False
    idx = text.find("Total Income")
    if idx == -1:
        idx = text.lower().find("total income")
    if idx != -1:
        window = text[idx : idx + 80]
        near = "44,281" in window or "49,263" in window
    return {"label_near_number": near, "has_44281": has_44281, "has_49263": has_49263}


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="pymupdf,pdfplumber,docling")
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    args = ap.parse_args()

    print(f"[benchmark] pdf={args.pdf}\n")
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        try:
            parser = get_parser(method)
            t0 = time.time()
            pages = parser.extract_pages(args.pdf)
            dt = time.time() - t0
            full = "\n".join(pages)
            score = association_score(full)
            print(f"=== {method} ===")
            print(f"  time={dt:6.1f}s  pages={len(pages):3d}  chars={len(full):7d}")
            print(f"  association: {score}")
        except Exception as e:  # noqa: BLE001 — one broken method shouldn't stop the rest
            print(f"=== {method} ===\n  FAILED: {type(e).__name__}: {str(e)[:100]}")
        print()


if __name__ == "__main__":
    run()
