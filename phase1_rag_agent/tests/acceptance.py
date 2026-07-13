"""Runs the 5 mandated acceptance scenarios and prints a graded report.

Usage: python -m tests.acceptance [--pdf PATH]
Exit code is 0 always (this is a report, not a hard gate) — read the output.
"""

import argparse
import sys
import time

from src.agent import NOT_FOUND, RAGAgent
from src.config import CONFIG
from src.index_store import get_or_build_index

DEFAULT_PDF = "data/earnings_presentation_q2fy26.pdf"


def has_citation(ans: str) -> bool:
    import re

    return bool(re.search(r"\[p\d+", ans))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    args = parser.parse_args()

    print(f"[acceptance] provider={CONFIG.llm_provider} pdf={args.pdf}")
    index = get_or_build_index(args.pdf, CONFIG)
    agent = RAGAgent(index=index, config=CONFIG)

    # Space out calls on hosted free tiers to respect per-minute rate limits.
    spacing = 5.0 if CONFIG.llm_provider == "gemini" else 0.0

    def ask(q):
        t0 = time.time()
        turn = agent.ask(q)
        dt = time.time() - t0
        print(
            f"\n{'=' * 70}\nQ: {q}\nA: {turn.answer}\n[{dt:.1f}s | retrieved: "
            f"{[r.chunk.citation for r in turn.retrieved]}]"
        )
        if spacing:
            time.sleep(spacing)
        return turn

    results = []

    # 1. Grounded fact
    t = ask("What are the major business segments discussed in the document?")
    results.append(("grounded-fact", t.answer != NOT_FOUND and has_citation(t.answer)))

    # 2. Numeric
    t = ask("What is the consolidated total income in H1-26?")
    ok_num = has_citation(t.answer) and (
        "44,281" in t.answer or "49,263" in t.answer or t.answer == NOT_FOUND
    )
    results.append(("numeric", ok_num))

    # 3. Cross-section
    t = ask("What drivers are mentioned for EBITDA changes in H1-26?")
    results.append(("cross-section", t.answer != NOT_FOUND and has_citation(t.answer)))

    # 4. Negative control
    t = ask("What is the CEO's email address?")
    results.append(("negative-control", t.answer == NOT_FOUND))

    # 5. Conversational follow-up
    ask("Summarize airport performance in H1-26.")
    t = ask("Break that down into passenger and cargo changes.")
    results.append(("follow-up", t.answer != NOT_FOUND and has_citation(t.answer)))

    print(f"\n{'=' * 70}\nSCORECARD")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
