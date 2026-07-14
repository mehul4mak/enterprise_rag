"""Scoring for RAG answers: deterministic checks + a combined LLM-as-judge.

Two kinds of signal:
  * Deterministic (free, no LLM): refusal correctness, citation validity, required-fact presence.
  * LLM-as-judge (1 call/question): faithfulness (grounded in context?) + answer relevance, scored
    together in one structured JSON response to keep token/cost low.

GCP mapping: the judge is the same pattern as Vertex AI's Gen AI Evaluation Service (pointwise
metrics with a model-based rater).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..config import Config
from ..grounding import CITATION_RE, valid_citations
from ..llm import generate
from ..retriever import RetrievedChunk

# --------------------------------------------------------------------- deterministic checks


def is_refusal(answer: str) -> bool:
    return answer.strip().lower().startswith("not found")


def citations_are_valid(answer: str, retrieved: list[RetrievedChunk]) -> bool:
    """Every citation in the answer must point at a chunk that was actually retrieved."""
    cited = CITATION_RE.findall(answer)
    if not cited:
        return False
    valid = valid_citations(retrieved)
    return all(c in valid for c in cited)


def required_facts_present(answer: str, must_include: list[str]) -> bool:
    return all(tok in answer for tok in must_include)


# --------------------------------------------------------------------- LLM-as-judge

_JUDGE_SYSTEM = """You are a strict evaluator of answers produced by a document-grounded RAG system.
You are given a QUESTION, the CONTEXT passages the system retrieved, and the system's ANSWER.
Score only from the CONTEXT and ANSWER — do not use outside knowledge.

Return ONLY a compact JSON object with these keys:
  "faithfulness": float 0..1   (1 = every claim in the ANSWER is supported by the CONTEXT; 0 = fabricated)
  "answer_relevance": float 0..1  (1 = directly answers the QUESTION; 0 = off-topic)
  "citation_supported": 0 or 1  (1 = the cited passages actually contain the claimed facts)
  "reason": short string (<= 25 words)
No prose outside the JSON."""


def _build_judge_prompt(question: str, answer: str, retrieved: list[RetrievedChunk]) -> str:
    context = "\n\n".join(f"{r.chunk.citation} {r.chunk.text}" for r in retrieved)
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer}\n\nJSON:"


def _extract_json(text: str) -> dict:
    """Robustly pull the first JSON object out of a model response."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


@dataclass
class JudgeScores:
    faithfulness: float
    answer_relevance: float
    citation_supported: float
    reason: str


def judge_answer(
    question: str, answer: str, retrieved: list[RetrievedChunk], config: Config
) -> JudgeScores:
    """One LLM call → faithfulness + answer_relevance + citation_supported."""
    prompt = _build_judge_prompt(question, answer, retrieved)
    raw = generate(prompt, _JUDGE_SYSTEM, config)
    data = _extract_json(raw)

    def _f(key: str) -> float:
        try:
            return max(0.0, min(1.0, float(data.get(key, 0.0))))
        except (TypeError, ValueError):
            return 0.0

    return JudgeScores(
        faithfulness=_f("faithfulness"),
        answer_relevance=_f("answer_relevance"),
        citation_supported=_f("citation_supported"),
        reason=str(data.get("reason", ""))[:200],
    )
