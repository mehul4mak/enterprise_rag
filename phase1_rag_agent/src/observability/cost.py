"""Token + cost estimation (local stand-in for Cloud Billing / per-request cost attribution).

We don't get exact token counts back from every backend, so we *estimate* (≈4 chars/token) and
price with a small table. Good enough for a per-stage cost dashboard and for optimization decisions;
documented as an estimate, not a bill.

GCP mapping: Cloud Billing export + per-request cost labels; Vertex returns usage_metadata you'd use
instead of the heuristic once on Vertex.
"""

from __future__ import annotations

# USD per 1M tokens: (input, output). Local models are free.
PRICING: dict[str, tuple[float, float]] = {
    # Gemini (Developer API / Vertex) — flash-lite tier
    "gemini-flash-lite-latest": (0.10, 0.40),
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-2.0-flash-001": (0.10, 0.40),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    # Anthropic (Haiku-class)
    "claude-haiku": (0.80, 4.00),
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token)."""
    return max(1, len(text) // 4)


def price_for(model: str) -> tuple[float, float]:
    return PRICING.get(model, (0.0, 0.0))


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = price_for(model)
    return round((input_tokens * pin + output_tokens * pout) / 1_000_000, 8)


def estimate_call_cost(model: str, prompt: str, completion: str) -> dict:
    """Token + cost estimate for one LLM call, as span/metric attributes."""
    itok = estimate_tokens(prompt)
    otok = estimate_tokens(completion)
    return {
        "input_tokens": itok,
        "output_tokens": otok,
        "cost_usd": cost_usd(model, itok, otok),
    }
