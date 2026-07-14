"""LangGraph RAG agent with the full governance/observability layer (Phase 2+).

Topology:

    START
      -> input_guard        (Model Armor: prompt-injection/jailbreak screen)
           |-- blocked --> finalize            (safe refusal; pipeline skipped)
           `-- allowed --> condense
      condense -> retrieve -> generate -> validate  (grounding/citation check)
      validate -> output_guard   (Model Armor: PII/secret redaction)
      output_guard -> finalize   (emit lineage + metrics + trace)
      finalize -> END

Every node runs inside a trace span (per-stage latency), feeds the metrics registry, and emits a
structured log line. Depends only on the provider interfaces, so the GCP backend needs no changes.

GCP mapping: nodes≈Cloud Trace spans, metrics≈Cloud Monitoring, logs≈Cloud Logging,
finalize≈Dataplex lineage sink, guards≈Model Armor/DLP.
"""

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import Turn
from .config import Config
from .governance import guardrails
from .governance.lineage import build_record, record_lineage
from .grounding import postprocess_answer
from .observability.cost import estimate_call_cost
from .observability.logging_setup import get_logger, log_event
from .observability.metrics import METRICS
from .observability.tracing import Trace, export_otel
from .prompts import (
    SYSTEM_CONDENSE,
    SYSTEM_GROUNDED,
    build_condense_prompt,
    build_qa_prompt,
)
from .providers.base import LLMProvider, Retriever
from .retriever import RetrievedChunk

_LOG = get_logger()


class GraphState(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]
    search_query: str
    retrieved: list[RetrievedChunk]
    answer: str
    blocked: bool
    guard_findings: list
    trace: Trace


def _record_cost(span, config: Config, prompt: str, completion: str) -> None:
    """Attach token/cost estimates to an LLM span and to process metrics."""
    est = estimate_call_cost(config.active_model, prompt, completion)
    span.set(**est)
    METRICS.incr("llm.input_tokens", est["input_tokens"])
    METRICS.incr("llm.output_tokens", est["output_tokens"])
    # cost is fractional; track in micro-USD to keep the counter integer-friendly
    METRICS.incr("llm.cost_micro_usd", int(round(est["cost_usd"] * 1_000_000)))


def build_graph(retriever: Retriever, llm: LLMProvider, config: Config, source_document: str = ""):
    """Compile the governed RAG StateGraph over the given providers."""

    def input_guard(state: GraphState) -> GraphState:
        trace: Trace = state["trace"]
        with trace.span("input_guard") as sp:
            decision = guardrails.input_guard(state["question"], config)
            sp.set(action=decision.action, findings=len(decision.findings))
            findings = [f.category for f in decision.findings]
            if not decision.allowed:
                METRICS.incr("guardrail.input.blocked")
                return {"blocked": True, "answer": decision.text, "guard_findings": findings}
            return {"blocked": False, "guard_findings": findings}

    def condense(state: GraphState) -> GraphState:
        trace: Trace = state["trace"]
        with trace.span("condense") as sp:
            question = state["question"]
            history = state.get("history") or []
            if not history:
                sp.set(rewritten=False)
                return {"search_query": question}
            prompt = build_condense_prompt(history[-config.max_history_turns :], question)
            try:
                rewritten = llm.generate(prompt, SYSTEM_CONDENSE).strip()
            except Exception:  # noqa: BLE001
                sp.set(rewritten=False, error=True)
                return {"search_query": question}
            _record_cost(sp, config, SYSTEM_CONDENSE + prompt, rewritten)
            if not rewritten or len(rewritten) > 300:
                sp.set(rewritten=False)
                return {"search_query": question}
            sp.set(rewritten=True)
            return {"search_query": rewritten}

    def retrieve_node(state: GraphState) -> GraphState:
        trace: Trace = state["trace"]
        with trace.span("retrieve") as sp:
            chunks = retriever.retrieve(state["search_query"], config.top_k_final)
            sp.set(
                k=len(chunks),
                citations=[c.chunk.citation for c in chunks],
                top_score=round(chunks[0].display_score, 4) if chunks else None,
            )
            return {"retrieved": chunks}

    def generate_node(state: GraphState) -> GraphState:
        trace: Trace = state["trace"]
        with trace.span("generate") as sp:
            retrieved = state["retrieved"]
            context_blocks = [f"{r.chunk.citation} {r.chunk.text}" for r in retrieved]
            prompt = build_qa_prompt(state["question"], context_blocks)
            answer = llm.generate(prompt, SYSTEM_GROUNDED)
            sp.set(chars=len(answer))
            _record_cost(sp, config, SYSTEM_GROUNDED + prompt, answer)
            return {"answer": answer}

    def validate_node(state: GraphState) -> GraphState:
        trace: Trace = state["trace"]
        with trace.span("validate") as sp:
            grounded = postprocess_answer(state["answer"], state["retrieved"])
            refused = grounded.strip().lower().startswith("not found")
            sp.set(refused=refused)
            if refused:
                METRICS.incr("answer.refused")
            return {"answer": grounded}

    def output_guard(state: GraphState) -> GraphState:
        trace: Trace = state["trace"]
        with trace.span("output_guard") as sp:
            decision = guardrails.output_guard(state["answer"], config)
            sp.set(action=decision.action, findings=len(decision.findings))
            if decision.action == "redact":
                METRICS.incr("guardrail.output.redacted")
            existing = state.get("guard_findings") or []
            return {
                "answer": decision.text,
                "guard_findings": existing + [f.category for f in decision.findings],
            }

    def finalize(state: GraphState) -> GraphState:
        trace: Trace = state["trace"]
        with trace.span("finalize"):
            retrieved = state.get("retrieved") or []
            record = build_record(
                trace_id=trace.trace_id,
                backend=config.backend,
                llm_provider=config.llm_provider,
                source_document=source_document,
                question=state["question"],
                search_query=state.get("search_query", state["question"]),
                retrieved=retrieved,
                answer=state["answer"],
            )
            record_lineage(record)
        METRICS.record_trace(trace)
        export_otel(trace)
        cost_usd = round(sum(sp.attributes.get("cost_usd", 0.0) for sp in trace.spans), 8)
        log_event(
            _LOG,
            "rag.query.complete",
            trace_id=trace.trace_id,
            backend=config.backend,
            provider=config.llm_provider,
            blocked=state.get("blocked", False),
            refused=record.refused,
            latencies=trace.latencies(),
            cost_usd=cost_usd,
            citations=record.cited,
        )
        return {}

    def route_after_input(state: GraphState) -> str:
        return "finalize" if state.get("blocked") else "condense"

    g = StateGraph(GraphState)
    for name, fn in [
        ("input_guard", input_guard),
        ("condense", condense),
        ("retrieve", retrieve_node),
        ("generate", generate_node),
        ("validate", validate_node),
        ("output_guard", output_guard),
        ("finalize", finalize),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "input_guard")
    g.add_conditional_edges(
        "input_guard", route_after_input, {"condense": "condense", "finalize": "finalize"}
    )
    g.add_edge("condense", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "validate")
    g.add_edge("validate", "output_guard")
    g.add_edge("output_guard", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


@dataclass
class RAGGraphAgent:
    """Multi-turn wrapper around the governed graph. API-compatible with Phase 1 RAGAgent."""

    retriever: Retriever
    llm: LLMProvider
    config: Config
    source_document: str = ""
    history: list[Turn] = field(default_factory=list)

    def __post_init__(self):
        self._graph = build_graph(self.retriever, self.llm, self.config, self.source_document)

    def ask(self, question: str) -> Turn:
        hist = [(t.question, t.answer) for t in self.history[-self.config.max_history_turns :]]

        # Semantic cache: only for standalone (no-history) questions — follow-ups depend on history.
        cacheable = self.config.semantic_cache and not hist
        qvec = self._query_embedding(question) if cacheable else None
        if qvec is not None:
            cached = self._cache_lookup(question, qvec)
            if cached is not None:
                self.history.append(cached)
                return cached

        trace = Trace()
        result = self._graph.invoke({"question": question, "history": hist, "trace": trace})
        turn = Turn(
            question=question,
            answer=result["answer"],
            retrieved=result.get("retrieved", []),
            trace=trace.to_dict(),
            guard_findings=result.get("guard_findings") or [],
            blocked=result.get("blocked", False),
        )
        self.history.append(turn)
        if qvec is not None and not turn.blocked:
            from .optimize.semantic_cache import CACHE

            CACHE.put(self.source_document, qvec, turn.answer, turn.retrieved)
        return turn

    def _query_embedding(self, question: str):
        from .embeddings import embed_texts

        return embed_texts([question], self.config.embedding_model)[0]

    def _cache_lookup(self, question: str, qvec) -> Turn | None:
        from .optimize.semantic_cache import CACHE

        hit = CACHE.get(self.source_document, qvec, self.config.cache_threshold)
        if hit is None:
            METRICS.incr("cache.miss")
            return None
        answer, retrieved = hit
        METRICS.incr("cache.hit")
        return Turn(
            question=question,
            answer=answer,
            retrieved=retrieved,
            trace={"cache_hit": True, "total_latency_ms": 0.0, "spans": []},
            guard_findings=[],
            blocked=False,
        )
