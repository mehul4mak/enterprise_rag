"""Eval-harness unit tests — deterministic parts only (no LLM calls)."""

from src.eval import judges
from src.eval.harness import _aggregate, load_dataset, to_markdown
from src.ingest import Chunk
from src.retriever import RetrievedChunk


def _rc(page=22, cid="c35", text="TOTAL INCOME 44,281"):
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, page=page, text=text),
        dense_score=1.0,
        sparse_score=1.0,
        fused_score=1.0,
    )


def test_refusal_detection():
    assert judges.is_refusal("Not found in the document.")
    assert not judges.is_refusal("The total income is 44,281 [p22:c35].")


def test_citation_validity_and_facts():
    rc = [_rc()]
    assert judges.citations_are_valid("It was 44,281 [p22:c35].", rc)
    assert not judges.citations_are_valid("It was X [p99:c9].", rc)  # hallucinated cite
    assert not judges.citations_are_valid("No citation here.", rc)
    assert judges.required_facts_present("It was 44,281 crore", ["44,281"])
    assert not judges.required_facts_present("It was 40,000", ["44,281"])


def test_judge_json_extraction_is_robust():
    d = judges._extract_json('junk {"faithfulness": 0.9, "answer_relevance": 1} trailing')
    assert d["faithfulness"] == 0.9


def test_dataset_loads_and_has_negative_controls():
    rows = load_dataset()
    assert len(rows) >= 5
    assert any(r["expect"] == "refuse" for r in rows)
    assert any(r["expect"] == "answer" for r in rows)


def test_aggregate_gates_math():
    results = [
        {
            "id": "a",
            "expect": "answer",
            "answered": True,
            "behaviour_correct": True,
            "citations_valid": True,
            "facts_present": True,
            "faithfulness": 0.9,
            "answer_relevance": 0.9,
            "citation_supported": 1.0,
        },
        {"id": "b", "expect": "refuse", "answered": False, "behaviour_correct": True},
    ]
    m = _aggregate(results, use_judge=True)
    assert m["behaviour_accuracy"] == 1.0
    assert m["citation_validity"] == 1.0
    assert m["all_gates_pass"] is True
    # markdown renders without error
    assert "RAG Evaluation Report" in to_markdown({"metrics": m, "results": results})


def test_aggregate_flags_wrong_behaviour():
    results = [
        {
            "id": "neg",
            "expect": "refuse",
            "answered": True,
            "behaviour_correct": False,
            "citations_valid": True,
            "facts_present": True,
            "faithfulness": 0.9,
            "answer_relevance": 0.9,
            "citation_supported": 1.0,
        },
    ]
    m = _aggregate(results, use_judge=True)
    assert m["behaviour_accuracy"] == 0.0
    assert m["all_gates_pass"] is False
