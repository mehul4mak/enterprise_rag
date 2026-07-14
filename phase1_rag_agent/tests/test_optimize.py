"""Phase 4B tests: cost estimation + semantic cache (deterministic, no LLM)."""

import numpy as np

from src.observability.cost import cost_usd, estimate_call_cost, estimate_tokens
from src.optimize.semantic_cache import SemanticCache


# ---------------- cost ----------------
def test_token_estimate_scales_with_length():
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("") == 1  # floor


def test_cost_zero_for_local_model():
    assert cost_usd("gemma2:2b", 1000, 1000) == 0.0


def test_cost_nonzero_for_priced_model():
    c = cost_usd("gemini-flash-lite-latest", 1_000_000, 1_000_000)
    assert c == round(0.10 + 0.40, 8)  # $0.50


def test_estimate_call_cost_shape():
    est = estimate_call_cost("gemini-flash-lite-latest", "hello world " * 50, "answer " * 20)
    assert set(est) == {"input_tokens", "output_tokens", "cost_usd"}
    assert est["input_tokens"] > est["output_tokens"]


# ---------------- semantic cache ----------------
def _vec(*xs):
    v = np.array(xs, dtype="float32")
    return v / np.linalg.norm(v)


def test_cache_hit_on_near_duplicate():
    c = SemanticCache()
    c.put("doc1", _vec(1, 0, 0), "cached answer [p1:c1]", ["chunk"])
    hit = c.get("doc1", _vec(0.99, 0.01, 0), threshold=0.97)
    assert hit is not None and hit[0] == "cached answer [p1:c1]"


def test_cache_miss_below_threshold():
    c = SemanticCache()
    c.put("doc1", _vec(1, 0, 0), "a", [])
    assert c.get("doc1", _vec(0, 1, 0), threshold=0.97) is None


def test_cache_is_isolated_per_document():
    c = SemanticCache()
    c.put("docA", _vec(1, 0, 0), "answer A", [])
    assert c.get("docB", _vec(1, 0, 0), threshold=0.9) is None  # different doc → no leak
    assert c.size("docA") == 1 and c.size("docB") == 0
