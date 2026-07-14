"""LangGraph rebuild of the RAG agent (Phase 2).

The Phase 1 agent was a hand-rolled method chain. Here the same flow is an explicit
LangGraph StateGraph — condense -> retrieve -> generate -> validate — which gives:
  * a declarative, inspectable topology (matches the JD's LangGraph requirement),
  * clean seams for observability/guardrails, and
  * provider-agnostic nodes (they depend only on the Retriever/LLMProvider interfaces),
    so switching to the GCP backend needs no graph changes.
"""

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import Turn
from .config import Config
from .grounding import postprocess_answer
from .prompts import (
    SYSTEM_CONDENSE,
    SYSTEM_GROUNDED,
    build_condense_prompt,
    build_qa_prompt,
)
from .providers.base import LLMProvider, Retriever
from .retriever import RetrievedChunk


class GraphState(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]
    search_query: str
    retrieved: list[RetrievedChunk]
    answer: str


def build_graph(retriever: Retriever, llm: LLMProvider, config: Config):
    """Compile the RAG StateGraph over the given providers."""

    def condense(state: GraphState) -> GraphState:
        question = state["question"]
        history = state.get("history") or []
        if not history:
            return {"search_query": question}
        prompt = build_condense_prompt(history[-config.max_history_turns :], question)
        try:
            rewritten = llm.generate(prompt, SYSTEM_CONDENSE).strip()
        except Exception:  # noqa: BLE001 — condensation must never break the turn
            return {"search_query": question}
        if not rewritten or len(rewritten) > 300:
            return {"search_query": question}
        return {"search_query": rewritten}

    def retrieve_node(state: GraphState) -> GraphState:
        chunks = retriever.retrieve(state["search_query"], config.top_k_final)
        return {"retrieved": chunks}

    def generate_node(state: GraphState) -> GraphState:
        retrieved = state["retrieved"]
        context_blocks = [f"{r.chunk.citation} {r.chunk.text}" for r in retrieved]
        prompt = build_qa_prompt(state["question"], context_blocks)
        return {"answer": llm.generate(prompt, SYSTEM_GROUNDED)}

    def validate_node(state: GraphState) -> GraphState:
        return {"answer": postprocess_answer(state["answer"], state["retrieved"])}

    g = StateGraph(GraphState)
    g.add_node("condense", condense)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("validate", validate_node)
    g.add_edge(START, "condense")
    g.add_edge("condense", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "validate")
    g.add_edge("validate", END)
    return g.compile()


@dataclass
class RAGGraphAgent:
    """Multi-turn wrapper around the compiled graph. API-compatible with Phase 1 RAGAgent."""

    retriever: Retriever
    llm: LLMProvider
    config: Config
    history: list[Turn] = field(default_factory=list)

    def __post_init__(self):
        self._graph = build_graph(self.retriever, self.llm, self.config)

    def ask(self, question: str) -> Turn:
        hist = [(t.question, t.answer) for t in self.history[-self.config.max_history_turns :]]
        result = self._graph.invoke({"question": question, "history": hist})
        turn = Turn(
            question=question,
            answer=result["answer"],
            retrieved=result.get("retrieved", []),
        )
        self.history.append(turn)
        return turn
