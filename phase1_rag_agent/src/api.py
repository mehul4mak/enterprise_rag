"""FastAPI service exposing the RAG agent over HTTP so it can be hosted.

Endpoints:
    GET  /health           liveness/readiness probe
    POST /ingest           {"pdf_path": "..."}          (re)build/load the active index
    POST /chat             {"question": "...", "session_id": "..."}  grounded answer + citations
    GET  /sessions/{id}    conversation history for a session
    DELETE /sessions/{id}  reset a session

Run: uvicorn src.api:app --host 0.0.0.0 --port 8000
Set INDEX_PDF=/path/to.pdf to auto-load a document at startup.
"""

import os
import re
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import CONFIG
from .graph import RAGGraphAgent
from .pipeline import index_document, new_agent
from .providers.base import Retriever

_CITATION_RE = re.compile(r"\[p\d+(?::c\d+)?\]")


class AppState:
    """Holds the active indexed retriever and per-session graph agents (in-memory)."""

    retriever: Retriever | None = None
    pdf_path: str | None = None
    sessions: dict[str, RAGGraphAgent] = {}


state = AppState()


def _load_index(pdf_path: str) -> None:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    state.retriever = index_document(pdf_path, CONFIG)
    state.pdf_path = pdf_path
    state.sessions.clear()  # index changed → old histories are stale


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_pdf = os.getenv("INDEX_PDF")
    if startup_pdf:
        _load_index(startup_pdf)
    yield
    state.sessions.clear()


app = FastAPI(
    title="Enterprise RAG",
    description="Document-grounded conversational RAG with citations and refusal.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- schemas ----------
class IngestRequest(BaseModel):
    pdf_path: str = Field(..., description="Server-local path to a PDF to index.")


class IngestResponse(BaseModel):
    pdf_path: str
    chunks: int
    pages: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = Field(None, description="Omit to start a new session.")


class RetrievedItem(BaseModel):
    citation: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[str]
    retrieved: list[RetrievedItem]
    blocked: bool = False
    guard_findings: list[str] = []
    latency_ms: dict[str, float] = {}
    total_latency_ms: float = 0.0
    trace_id: str | None = None


# ---------- routes ----------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "backend": CONFIG.backend,
        "provider": CONFIG.llm_provider,
        "index_loaded": state.retriever is not None,
        "pdf_path": state.pdf_path,
        "active_sessions": len(state.sessions),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    try:
        _load_index(req.pdf_path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    assert state.retriever is not None
    chunks = state.retriever.all_chunks()
    return IngestResponse(
        pdf_path=req.pdf_path,
        chunks=len(chunks),
        pages=len(set(c.page for c in chunks)),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if state.retriever is None:
        raise HTTPException(status_code=409, detail="No document indexed. POST /ingest first.")

    session_id = req.session_id or uuid.uuid4().hex
    agent = state.sessions.get(session_id)
    if agent is None:
        agent = new_agent(state.retriever, CONFIG, source_document=state.pdf_path or "")
        state.sessions[session_id] = agent

    try:
        turn = agent.ask(req.question)
    except Exception as e:  # noqa: BLE001 — surface LLM/backend errors as 502
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}") from e

    trace = turn.trace or {}
    return ChatResponse(
        session_id=session_id,
        answer=turn.answer,
        citations=_CITATION_RE.findall(turn.answer),
        retrieved=[
            RetrievedItem(
                citation=r.chunk.citation,
                score=round(r.display_score, 4),
                snippet=" ".join(r.chunk.text.split())[:200],
            )
            for r in turn.retrieved
        ],
        blocked=turn.blocked,
        guard_findings=turn.guard_findings or [],
        latency_ms={sp["name"]: sp["latency_ms"] for sp in trace.get("spans", [])},
        total_latency_ms=trace.get("total_latency_ms", 0.0),
        trace_id=trace.get("trace_id"),
    )


@app.get("/metrics")
def metrics() -> dict:
    """Cloud Monitoring-style snapshot: per-stage latency summaries + counters."""
    from .observability.metrics import METRICS

    return METRICS.snapshot()


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    agent = state.sessions.get(session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    return {
        "session_id": session_id,
        "turns": [{"question": t.question, "answer": t.answer} for t in agent.history],
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    existed = state.sessions.pop(session_id, None) is not None
    return {"session_id": session_id, "deleted": existed}
