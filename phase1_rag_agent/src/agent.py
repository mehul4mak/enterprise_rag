"""Conversational RAG agent: multi-turn, grounded, cited, with refusal logic."""

from dataclasses import dataclass, field

from .config import Config
from .grounding import NOT_FOUND, postprocess_answer
from .index_store import HybridIndex
from .llm import generate
from .prompts import (
    SYSTEM_CONDENSE,
    SYSTEM_GROUNDED,
    build_condense_prompt,
    build_qa_prompt,
)
from .retriever import RetrievedChunk, retrieve

__all__ = ["NOT_FOUND", "Turn", "RAGAgent"]


@dataclass
class Turn:
    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    # Optional governance/observability metadata (populated by the LangGraph agent).
    trace: dict | None = None
    guard_findings: list | None = None
    blocked: bool = False


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

    def _postprocess(self, answer: str, retrieved: list[RetrievedChunk]) -> str:
        return postprocess_answer(answer, retrieved)

    def ask(self, question: str) -> Turn:
        search_query = self._condense(question)
        retrieved = retrieve(search_query, self.index, self.config)

        context_blocks = [f"{r.chunk.citation} {r.chunk.text}" for r in retrieved]
        prompt = build_qa_prompt(question, context_blocks)
        raw = generate(prompt, SYSTEM_GROUNDED, self.config)
        answer = postprocess_answer(raw, retrieved)

        turn = Turn(question=question, answer=answer, retrieved=retrieved)
        self.history.append(turn)
        return turn
