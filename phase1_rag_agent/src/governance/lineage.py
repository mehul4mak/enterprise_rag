"""Data lineage tracking (local stand-in for Dataplex Lineage).

For every answered query we record the full provenance chain:

    source document  ->  chunks retrieved (with scores)  ->  chunks actually cited  ->  answer

This is the "assured lineage around data used by agent" the JD calls for. Records are appended as
JSONL so they can be shipped to BigQuery/Dataplex in Phase 3.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import CACHE_DIR

_CITATION_RE = re.compile(r"\[p\d+(?::c\d+)?\]")
_LINEAGE_PATH = CACHE_DIR / "lineage.jsonl"


@dataclass
class LineageRecord:
    trace_id: str
    timestamp: float
    backend: str
    llm_provider: str
    source_document: str
    question: str
    search_query: str
    retrieved: list[dict]  # [{citation, page, chunk_id, score}]
    cited: list[str]  # citations that actually appear in the answer
    answer_citations_valid: bool
    refused: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def build_record(
    *,
    trace_id: str,
    backend: str,
    llm_provider: str,
    source_document: str,
    question: str,
    search_query: str,
    retrieved,
    answer: str,
) -> LineageRecord:
    cited = _CITATION_RE.findall(answer)
    retrieved_meta = [
        {
            "citation": r.chunk.citation,
            "page": r.chunk.page,
            "chunk_id": r.chunk.chunk_id,
            "score": round(r.display_score, 4),
        }
        for r in retrieved
    ]
    return LineageRecord(
        trace_id=trace_id,
        timestamp=time.time(),
        backend=backend,
        llm_provider=llm_provider,
        source_document=source_document,
        question=question,
        search_query=search_query,
        retrieved=retrieved_meta,
        cited=cited,
        answer_citations_valid=bool(cited),
        refused=answer.strip().lower().startswith("not found"),
    )


def record_lineage(record: LineageRecord, path: Path = _LINEAGE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(record.to_json() + "\n")
