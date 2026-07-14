"""RAG evaluation harness — runs the agent over a golden dataset and scores it.

Per question it computes:
  * behaviour_correct   — did it answer vs refuse as expected? (deterministic)
  * citations_valid     — do citations point at retrieved chunks? (deterministic, answers only)
  * facts_present       — are required gold facts in the answer? (deterministic, answers only)
  * faithfulness        — grounded in context, no fabrication (LLM-judge, answers only)
  * answer_relevance    — actually answers the question (LLM-judge, answers only)
  * citation_supported  — cited passages contain the claim (LLM-judge, answers only)

Aggregate metrics are gated (see GATES). Writes a JSON + Markdown report.

Usage:
    python -m src.eval.harness --pdf data/earnings_presentation_q2fy26.pdf
    python -m src.eval.harness --no-judge     # deterministic-only (zero LLM judge calls)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..config import CONFIG
from ..pipeline import build_agent
from . import judges

DATASET = Path(__file__).parent / "dataset.jsonl"

# Aggregate pass/fail gates (means over answered questions unless noted).
GATES = {
    "behaviour_accuracy": 1.0,  # every question must answer/refuse as expected
    "citation_validity": 1.0,  # every answered question must have valid citations
    "faithfulness_mean": 0.80,
    "answer_relevance_mean": 0.80,
}


def load_dataset(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(pdf: str, use_judge: bool = True) -> dict:
    agent = build_agent(pdf, CONFIG)
    rows = load_dataset()
    results = []
    spacing = 5.0 if CONFIG.llm_provider == "gemini" else 0.0

    for row in rows:
        t0 = time.time()
        turn = agent.ask(row["question"])
        answered = not judges.is_refusal(turn.answer)
        expected_answer = row["expect"] == "answer"
        behaviour_correct = answered == expected_answer

        rec = {
            "id": row["id"],
            "expect": row["expect"],
            "answered": answered,
            "behaviour_correct": behaviour_correct,
            "answer": turn.answer,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }

        if answered:
            rec["citations_valid"] = judges.citations_are_valid(turn.answer, turn.retrieved)
            rec["facts_present"] = judges.required_facts_present(
                turn.answer, row.get("must_include", [])
            )
            if use_judge:
                s = judges.judge_answer(row["question"], turn.answer, turn.retrieved, CONFIG)
                rec.update(
                    faithfulness=s.faithfulness,
                    answer_relevance=s.answer_relevance,
                    citation_supported=s.citation_supported,
                    judge_reason=s.reason,
                )
        results.append(rec)
        if spacing:
            time.sleep(spacing)

    return {"metrics": _aggregate(results, use_judge), "results": results}


def _mean(vals: list[float]) -> float:
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _aggregate(results: list[dict], use_judge: bool) -> dict:
    answered = [r for r in results if r["answered"]]
    metrics = {
        "n": len(results),
        "behaviour_accuracy": _mean([1.0 if r["behaviour_correct"] else 0.0 for r in results]),
        "citation_validity": _mean([1.0 if r.get("citations_valid") else 0.0 for r in answered]),
        "facts_accuracy": _mean([1.0 if r.get("facts_present") else 0.0 for r in answered]),
    }
    if use_judge and answered:
        metrics["faithfulness_mean"] = _mean([r.get("faithfulness", 0.0) for r in answered])
        metrics["answer_relevance_mean"] = _mean([r.get("answer_relevance", 0.0) for r in answered])
        metrics["citation_supported_mean"] = _mean(
            [r.get("citation_supported", 0.0) for r in answered]
        )
    metrics["gates"] = {
        k: {"threshold": thr, "value": metrics.get(k), "pass": (metrics.get(k) or 0.0) >= thr}
        for k, thr in GATES.items()
        if k in metrics
    }
    metrics["all_gates_pass"] = all(g["pass"] for g in metrics["gates"].values())
    return metrics


def to_markdown(report: dict) -> str:
    m = report["metrics"]
    lines = ["# RAG Evaluation Report", ""]
    lines.append(f"- questions: **{m['n']}**  ·  all gates pass: **{m['all_gates_pass']}**")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("| metric | value | gate | pass |")
    lines.append("|--------|------:|-----:|:----:|")
    for k in (
        "behaviour_accuracy",
        "citation_validity",
        "facts_accuracy",
        "faithfulness_mean",
        "answer_relevance_mean",
        "citation_supported_mean",
    ):
        if k in m:
            g = m["gates"].get(k)
            thr = g["threshold"] if g else "—"
            ok = "✅" if (g["pass"] if g else True) else "❌"
            lines.append(f"| {k} | {m[k]} | {thr} | {ok if g else ''} |")
    lines.append("")
    lines.append("## Per-question")
    lines.append("| id | expect | behaviour | faithful | relevance | cite_valid | latency ms |")
    lines.append("|----|--------|:---------:|:--------:|:---------:|:----------:|-----------:|")
    for r in report["results"]:
        lines.append(
            f"| {r['id']} | {r['expect']} | {'✅' if r['behaviour_correct'] else '❌'} "
            f"| {r.get('faithfulness', '—')} | {r.get('answer_relevance', '—')} "
            f"| {'✅' if r.get('citations_valid') else ('—' if not r['answered'] else '❌')} "
            f"| {r.get('latency_ms', '—')} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/earnings_presentation_q2fy26.pdf")
    ap.add_argument(
        "--no-judge", action="store_true", help="deterministic checks only (no LLM judge)"
    )
    ap.add_argument("--out", default="eval_report", help="output basename (writes .json + .md)")
    args = ap.parse_args()

    print(f"[eval] provider={CONFIG.llm_provider} judge={not args.no_judge} pdf={args.pdf}")
    report = evaluate(args.pdf, use_judge=not args.no_judge)

    Path(f"{args.out}.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    Path(f"{args.out}.md").write_text(md)
    print(md)
    print(f"\n[eval] wrote {args.out}.json and {args.out}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
