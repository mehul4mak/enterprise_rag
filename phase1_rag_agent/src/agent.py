"""Conversational RAG agent: multi-turn, grounded, cited, with refusal logic."""

import re
from dataclasses import dataclass, field

from .config import Config
from .index_store import HybridIndex
from .llm import generate
from .prompts import (
    SYSTEM_CONDENSE,
    SYSTEM_GROUNDED,
    build_condense_prompt,
    build_qa_prompt,
)
from .retriever import RetrievedChunk, retrieve

NOT_FOUND = "Not found in the document."
_CITATION_RE = re.compile(r"\[p\d+(?::c\d+)?\]")


@dataclass
class Turn:
    question: str
    answer: str
    retrieved: list[RetrievedChunk]


@dataclass
class RAGAgent:
    index: HybridIndex
    config: Config
    history: list[Turn] = field(default_factory=list)

    def _condense(self, question: str) -> str:
        """Rewrite a follow-up into a standalone query using recent history."""
        if not self.history:
            return question
        recent = [(t.question, t.answer) for t in self.history[-self.config.max_history_turns :]]
        prompt = build_condense_prompt(recent, question)
        try:
            rewritten = generate(prompt, SYSTEM_CONDENSE, self.config).strip()
        except Exception:  # noqa: BLE001 — never let condensation break the turn
            return question
        # Guard against a chatty model that ignored instructions.
        if not rewritten or len(rewritten) > 300:
            return question
        return rewritten

    def _valid_citations(self, retrieved: list[RetrievedChunk]) -> set[str]:
        valid = set()
        for r in retrieved:
            valid.add(f"[p{r.chunk.page}:{r.chunk.chunk_id}]")
            valid.add(f"[p{r.chunk.page}]")
        return valid

    def _postprocess(self, answer: str, retrieved: list[RetrievedChunk]) -> str:
        """Enforce grounding: an answer must either be the refusal or carry a valid citation."""
        answer = answer.strip()
        if answer.lower().startswith("not found"):
            return NOT_FOUND

        cited = _CITATION_RE.findall(answer)
        if not cited:
            # Model gave prose with no citation → treat as ungrounded.
            return NOT_FOUND

        valid = self._valid_citations(retrieved)
        # If none of the emitted citations correspond to retrieved chunks, it's hallucinated.
        if not any(c in valid for c in cited):
            return NOT_FOUND

        return answer

    def ask(self, question: str) -> Turn:
        search_query = self._condense(question)
        retrieved = retrieve(search_query, self.index, self.config)

        context_blocks = [
            f"{r.chunk.citation} {r.chunk.text}" for r in retrieved
        ]
        prompt = build_qa_prompt(question, context_blocks)
        raw = generate(prompt, SYSTEM_GROUNDED, self.config)
        answer = self._postprocess(raw, retrieved)

        turn = Turn(question=question, answer=answer, retrieved=retrieved)
        self.history.append(turn)
        return turn
