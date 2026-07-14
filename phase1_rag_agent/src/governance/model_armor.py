"""Model Armor — prompt/response screening (local stand-in for GCP Model Armor + DLP).

Screens:
  * INPUT  — prompt-injection / jailbreak / system-prompt-exfiltration attempts.
  * OUTPUT — PII / secret leakage (emails, phones, cards, SSNs, API keys).

GCP mapping: Model Armor (prompt & response shields) + Sensitive Data Protection (DLP) for PII.
This local version is heuristic/regex-based — deliberately transparent and dependency-free — and is
structured so Phase 3 swaps the body of `screen_*` for a Model Armor API call behind the same
`ArmorFinding` result type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- prompt-injection / jailbreak signatures ---
_INJECTION_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|prior|above)\s+(instructions|prompts?)",
    r"disregard (the|all|any)?\s*(previous|prior|above|system)",
    r"forget (everything|all|your) (instructions|rules)",
    r"you are now\b",
    r"act as (if you are|an?)\b",
    r"reveal (your|the) (system|hidden) (prompt|instructions)",
    r"print (your|the) (system|initial) (prompt|instructions)",
    r"developer mode",
    r"do anything now|(\bDAN\b)",
    r"bypass (your|the|all) (rules|filters|guardrails|safety)",
]

# --- PII / secret signatures (DLP-style) ---
_PII_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "PHONE": r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]?){13,16}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "API_KEY": r"\b(?:AIza[0-9A-Za-z_\-]{20,}|AQ\.[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{20,})\b",
}


@dataclass
class ArmorFinding:
    category: str  # e.g. "PROMPT_INJECTION", "PII:EMAIL"
    severity: str  # "high" | "medium" | "low"
    detail: str


def screen_prompt(text: str) -> list[ArmorFinding]:
    findings: list[ArmorFinding] = []
    low = text.lower()
    for pat in _INJECTION_PATTERNS:
        m = re.search(pat, low)
        if m:
            findings.append(
                ArmorFinding("PROMPT_INJECTION", "high", f"matched pattern: {m.group(0)!r}")
            )
    return findings


def detect_pii(text: str) -> list[ArmorFinding]:
    findings: list[ArmorFinding] = []
    for label, pat in _PII_PATTERNS.items():
        for m in re.finditer(pat, text):
            findings.append(ArmorFinding(f"PII:{label}", "high", _mask(m.group(0))))
    return findings


def screen_response(text: str) -> list[ArmorFinding]:
    return detect_pii(text)


def redact(text: str) -> str:
    """Replace detected PII spans with typed placeholders (DLP de-identify style)."""
    out = text
    for label, pat in _PII_PATTERNS.items():
        out = re.sub(pat, f"[REDACTED:{label}]", out)
    return out


def _mask(value: str) -> str:
    v = value.strip()
    if len(v) <= 4:
        return "*" * len(v)
    return v[:2] + "*" * (len(v) - 4) + v[-2:]
