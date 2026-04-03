"""
Structured JSON logging for the agent system.

Every log record is emitted as a single JSON line with fields:
  {timestamp, level, trace_id, component, event, data}

Configuration is loaded from config/v1/logging.yaml.  Values can be
overridden at runtime by passing explicit arguments to get_logger().

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
from typing import Any, Dict, Optional

import yaml

_CONFIG_V1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "v1")
_LOGGING_YAML = os.path.join(_CONFIG_V1, "logging.yaml")

_trace_id: ContextVar[str] = ContextVar("trace_id", default="unset")


def set_trace_id(value: str) -> None:
    """Set the trace ID for the current async context (one UUID per turn)."""
    _trace_id.set(value)


def get_trace_id() -> str:
    return _trace_id.get()


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _load_logging_yaml(path: str = _LOGGING_YAML) -> Dict[str, Any]:
    """
    Load logging defaults from config/v1/logging.yaml.

    Returns an empty dict if the file is missing or unparseable so that
    get_logger() can still fall back to hardcoded defaults.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw.pop("version", None)
        return raw
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# One-time root logger configuration
# ---------------------------------------------------------------------------

_configured = False


def _configure(log_path: str, log_level: str) -> None:
    global _configured
    if _configured:
        return
    _configured = True

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

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


def get_logger(
    component: str,
    log_path: Optional[str] = None,
    log_level: Optional[str] = None,
) -> logging.Logger:
    """
    Return a named logger, configuring the root logger once on first call.

    Values resolve in this order:
      1. Explicit ``log_path`` / ``log_level`` arguments.
      2. config/v1/logging.yaml (``path`` / ``level`` keys).
      3. Hardcoded defaults (``./logs/app.jsonl`` / ``INFO``).

    Args:
        component: Logger name, typically ``__name__``.
        log_path:  Override the log file path from YAML/defaults.
        log_level: Override the log level from YAML/defaults.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    yaml_cfg = _load_logging_yaml()
    resolved_path = log_path or yaml_cfg.get("path", "./logs/app.jsonl")
    resolved_level = log_level or yaml_cfg.get("level", "INFO")
    _configure(resolved_path, resolved_level)
    return logging.getLogger(component)
