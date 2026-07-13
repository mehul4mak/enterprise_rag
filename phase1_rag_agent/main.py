"""Conversational RAG over a PDF — single-command entrypoint.

Usage:
    python main.py --pdf ./data/earnings_presentation_q2fy26.pdf
    python main.py --pdf ./doc.pdf --no-rerank --reindex

Then ask questions in the chat loop. Commands inside the loop:
    :debug   toggle retrieval-visibility panel (top-k snippets + scores)
    :quit    exit
"""

import argparse
import sys

from src.agent import RAGAgent
from src.config import CONFIG
from src.index_store import get_or_build_index


def print_debug(turn) -> None:
    print("\n  ┌─ retrieval (top-k) ─────────────────────────────────────")
    for r in turn.retrieved:
        score = r.display_score
        snippet = " ".join(r.chunk.text.split())[:110]
        print(f"  │ {r.chunk.citation:14s} score={score:7.3f}  {snippet}")
    print("  └─────────────────────────────────────────────────────────")


def main() -> int:
    parser = argparse.ArgumentParser(description="Conversational RAG over a PDF.")
    parser.add_argument("--pdf", required=True, help="Path to a local PDF file.")
    parser.add_argument("--reindex", action="store_true", help="Force rebuild of the index.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking.")
    parser.add_argument("--debug", action="store_true", help="Start with retrieval debug panel on.")
    args = parser.parse_args()

    print(f"[setup] provider={CONFIG.llm_provider} | model="
          f"{CONFIG.ollama_model if CONFIG.llm_provider == 'ollama' else '(see .env)'}")
    print(f"[setup] indexing {args.pdf} ...")
    try:
        index = get_or_build_index(args.pdf, CONFIG, force_reindex=args.reindex)
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    print(f"[setup] {len(index.chunks)} chunks indexed across "
          f"{len(set(c.page for c in index.chunks))} pages.\n")

    agent = RAGAgent(index=index, config=CONFIG)
    debug = args.debug
    print("Ask questions about the document. Type :debug to toggle retrieval view, :quit to exit.\n")

    while True:
        try:
            question = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in (":quit", ":q", "exit"):
            break
        if question == ":debug":
            debug = not debug
            print(f"[debug retrieval {'ON' if debug else 'OFF'}]\n")
            continue

        turn = agent.ask(question)
        if debug:
            print_debug(turn)
        print(f"\nbot › {turn.answer}\n")

    print("bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
