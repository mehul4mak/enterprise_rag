"""Guardrail orchestration — the input/output safety gates around the RAG pipeline.

  INPUT gate  : Model Armor prompt screening (injection/jailbreak). On a high-severity hit we
                short-circuit to a safe refusal (the pipeline never runs).
  OUTPUT gate : Model Armor response screening (PII/secret leakage) -> redact before returning.

Grounding/citation validation lives in src/grounding.py and runs regardless; these guards are the
safety layer on top. GCP mapping: Model Armor shields + DLP de-identification + Vertex safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from . import model_armor
from .model_armor import ArmorFinding

SAFE_REFUSAL = "I can't process that request."


@dataclass
class GuardDecision:
    allowed: bool
    action: str  # "allow" | "block" | "redact"
    text: str
    findings: list[ArmorFinding] = field(default_factory=list)


def input_guard(question: str, config: Config) -> GuardDecision:
    if not (config.guardrails_enabled and config.model_armor_enabled):
        return GuardDecision(allowed=True, action="allow", text=question)

    findings = model_armor.screen_prompt(question)
    if any(f.severity == "high" for f in findings):
        return GuardDecision(allowed=False, action="block", text=SAFE_REFUSAL, findings=findings)
    return GuardDecision(allowed=True, action="allow", text=question, findings=findings)


def output_guard(answer: str, config: Config) -> GuardDecision:
    if not (config.guardrails_enabled and config.model_armor_enabled):
        return GuardDecision(allowed=True, action="allow", text=answer)

    findings = model_armor.screen_response(answer)
    if findings:
        return GuardDecision(
            allowed=True, action="redact", text=model_armor.redact(answer), findings=findings
        )
    return GuardDecision(allowed=True, action="allow", text=answer, findings=findings)
