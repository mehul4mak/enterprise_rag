"""Shared grounding/citation-validation logic (used by both the Phase 1 agent and the graph)."""

import re

from .retriever import RetrievedChunk

NOT_FOUND = "Not found in the document."
CITATION_RE = re.compile(r"\[p\d+(?::c\d+)?\]")


def valid_citations(retrieved: list[RetrievedChunk]) -> set[str]:
    valid: set[str] = set()
    for r in retrieved:
        valid.add(f"[p{r.chunk.page}:{r.chunk.chunk_id}]")
        valid.add(f"[p{r.chunk.page}]")
    return valid


def postprocess_answer(answer: str, retrieved: list[RetrievedChunk]) -> str:
    """Enforce grounding: an answer must either be the refusal or carry a *valid* citation."""
    answer = answer.strip()
    if answer.lower().startswith("not found"):
        return NOT_FOUND

    cited = CITATION_RE.findall(answer)
    if not cited:
        return NOT_FOUND

    if not any(c in valid_citations(retrieved) for c in cited):
        return NOT_FOUND

    return answer
