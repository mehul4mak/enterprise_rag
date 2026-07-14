"""Data residency / sovereignty enforcement (local stand-in for GCP sovereign controls).

The JD stresses "secure, sovereign cloud" and "data sovereignty". This module enforces a
residency policy that decides whether a given LLM provider is allowed to process the document:

    DATA_RESIDENCY=strict    -> NO data may leave the machine. Only local providers (ollama)
                                are allowed; external APIs (gemini/openai/anthropic) are blocked.
    DATA_RESIDENCY=regional  -> cloud allowed but must be an in-region/sovereign endpoint
                                (Phase 3: Vertex in GCP_LOCATION). Flagged, not blocked, locally.
    DATA_RESIDENCY=off       -> no restriction (default for the public earnings-deck demo).

GCP mapping: Organization Policy (resource-location constraints) + VPC-SC perimeter + choosing a
sovereign Vertex region. Enforcing it in-app makes the guarantee explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config

_LOCAL_PROVIDERS = {"ollama"}
_EXTERNAL_PROVIDERS = {"gemini", "openai", "anthropic"}


class ResidencyViolation(RuntimeError):
    pass


@dataclass
class ResidencyDecision:
    allowed: bool
    policy: str
    provider: str
    reason: str


def evaluate(config: Config) -> ResidencyDecision:
    policy = getattr(config, "data_residency", "off")
    provider = config.llm_provider

    if policy == "strict" and provider in _EXTERNAL_PROVIDERS:
        return ResidencyDecision(
            allowed=False,
            policy=policy,
            provider=provider,
            reason=(
                f"DATA_RESIDENCY=strict forbids external egress, but LLM_PROVIDER='{provider}' "
                f"sends data off-box. Use LLM_PROVIDER=ollama (local) or a sovereign-region "
                f"Vertex endpoint (BACKEND=gcp in {config.gcp_location})."
            ),
        )

    if policy == "regional" and provider in _EXTERNAL_PROVIDERS and config.backend != "gcp":
        return ResidencyDecision(
            allowed=True,  # allowed but flagged
            policy=policy,
            provider=provider,
            reason=(
                f"regional policy: '{provider}' via the public API is not region-pinned. "
                f"Phase 3 should route through Vertex in {config.gcp_location}."
            ),
        )

    return ResidencyDecision(allowed=True, policy=policy, provider=provider, reason="ok")


def enforce(config: Config) -> ResidencyDecision:
    decision = evaluate(config)
    if not decision.allowed:
        raise ResidencyViolation(decision.reason)
    return decision
