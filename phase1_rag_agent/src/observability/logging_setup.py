"""Structured JSON logging (local stand-in for Cloud Logging).

One JSON object per event with trace correlation. GCP mapping: Cloud Logging structured entries
(the `trace` field is what correlates logs↔Cloud Trace in the console).
"""

from __future__ import annotations

import json
import logging
import sys

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)  # type: ignore[attr-defined]
        return json.dumps(payload, default=str)


def get_logger(name: str = "enterprise_rag") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _CONFIGURED = True
    return logger


def log_event(logger: logging.Logger, message: str, *, trace_id: str | None = None, **fields):
    """Emit one structured log line with optional trace correlation."""
    extra = dict(fields)
    if trace_id:
        extra["trace"] = trace_id
    logger.info(message, extra={"extra_fields": extra})
