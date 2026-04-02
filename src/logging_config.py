"""
Structured JSON logging for the agent system.

Every log record is emitted as a single JSON line with fields:
  {timestamp, level, trace_id, component, event, data}

Usage:
    from src.logging_config import get_logger, set_trace_id

    logger = get_logger(__name__)
    set_trace_id("some-uuid")
    logger.info("event name", extra={"data": {"key": "value"}})
"""

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict

_trace_id: ContextVar[str] = ContextVar("trace_id", default="unset")


def set_trace_id(value: str) -> None:
    """Set the trace ID for the current context (per-turn UUID)."""
    _trace_id.set(value)


def get_trace_id() -> str:
    return _trace_id.get()


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: Dict[str, Any] = getattr(record, "data", {})
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "trace_id": _trace_id.get(),
            "component": record.name,
            "event": record.getMessage(),
            "data": data,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


_configured = False


def _configure(log_path: str = "./logs/app.jsonl", log_level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    _configured = True

    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    formatter = _JsonFormatter()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.WARNING)
    root.addHandler(stderr_handler)


def get_logger(component: str, log_path: str = "./logs/app.jsonl", log_level: str = "INFO") -> logging.Logger:
    """Return a logger for the given component, configuring the root logger once."""
    _configure(log_path, log_level)
    return logging.getLogger(component)
