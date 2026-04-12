"""
Structured JSON logging for the agent system.

Every log record is emitted as a single JSON line with fields:
  {timestamp, level, trace_id, component, event, data}

Initialisation
--------------
Call ``init_session_logging(session_log_dir, log_level)`` once at startup
(after the session ID is known) to create per-component log files inside the
session directory:

    logs/session_{session_id}_{YYYY-MM-DD}/
        app.jsonl     ← orchestrator, hooks, client, main
        memory.jsonl  ← src.memory.*
        agent.jsonl   ← src.agents.*, src.tools.*
        metrics.jsonl ← written directly by PostSessionHook

If ``init_session_logging`` is never called (e.g. in unit tests), all loggers
fall back to Python's default behaviour: WARNING+ to stderr.

Usage:
    from src.logging_config import get_logger, set_trace_id, init_session_logging

    init_session_logging("/path/to/logs/session_abc_2026-04-05", log_level="INFO")

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
from typing import Dict, Optional, Union

import yaml

from src.constants import (
    DEFAULT_APP_LOG_FILENAME,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_LEVEL,
    DEFAULT_TRACE_ID,
    LOG_COMPONENT_FILES,
)

_CONFIG_V1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "v1")
_LOGGING_YAML = os.path.join(_CONFIG_V1, "logging.yaml")

_trace_id: ContextVar[str] = ContextVar("trace_id", default=DEFAULT_TRACE_ID)


def set_trace_id(value: str) -> None:
    """Set the trace ID for the current async context (one UUID per turn)."""
    _trace_id.set(value)


def get_trace_id() -> str:
    """Return the trace ID for the current async context.

    Returns:
        The active trace ID string, or ``"unset"`` if none has been set.
    """
    return _trace_id.get()


# ---------------------------------------------------------------------------
# YAML loader (level override only)
# ---------------------------------------------------------------------------


def _load_logging_yaml(path: str = _LOGGING_YAML) -> Dict[str, Union[str, int]]:
    """
    Load the logging YAML and return scalar values.

    Only ``level`` is meaningful here; ``path`` is no longer used (the session
    log dir is determined at runtime).  Returns built-in defaults on any error.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dict with at least a ``level`` key.
    """
    result: Dict[str, Union[str, int]] = {"level": DEFAULT_LOG_LEVEL}
    if not os.path.exists(path):
        return result
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw.pop("version", None)
        result.update({k: v for k, v in raw.items() if isinstance(v, (str, int))})
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Logging formatter that serialises each record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Serialise a log record to a JSON string.

        The emitted object always contains: ``timestamp`` (ISO-8601 UTC),
        ``level``, ``trace_id``, ``component``, ``event``, and ``data``.
        An optional ``exception`` key is appended when ``record.exc_info``
        is set.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON string with no trailing newline.
        """
        data: object = getattr(record, "data", {})
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
# Session-scoped logging initialisation
# ---------------------------------------------------------------------------

_configured = False


def _attach_file_handler(
    logger: logging.Logger,
    path: str,
    level: int,
    formatter: logging.Formatter,
) -> None:
    """Open a FileHandler on *path* and attach it to *logger*."""
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(level)
    logger.addHandler(fh)


def init_session_logging(
    session_log_dir: str,
    log_level: Optional[str] = None,
    strands_log_level: Optional[str] = None,
) -> None:
    """
    Configure per-component file logging for a single session.

    Must be called **once** after the session ID is known.  Subsequent calls
    are no-ops (idempotent guard).

    Creates *session_log_dir* if it does not exist, then attaches
    ``FileHandler`` instances as follows:

    +-----------------------+---------------------+------------------+
    | Logger namespace      | File                | propagate        |
    +=======================+=====================+==================+
    | ``src.memory.*``      | ``memory.jsonl``    | False            |
    | ``src.agents.*``      | ``agent.jsonl``     | False            |
    | ``src.tools.*``       | ``agent.jsonl``     | False            |
    | root (everything else)| ``app.jsonl``       | n/a (root)       |
    +-----------------------+---------------------+------------------+

    ``strands.*`` loggers propagate to root (app.jsonl) at *strands_log_level*,
    which defaults to ``WARNING`` to suppress verbose Strands internals.
    Set ``STRANDS_LOG_LEVEL=DEBUG`` in ``.env`` to trace Strands event loops.

    A ``WARNING+`` stderr handler is also added to the root logger for
    immediate operator visibility.

    Args:
        session_log_dir:   Absolute or relative path to the session log folder.
                           Created automatically if absent.
        log_level:         App logging level (e.g. ``"DEBUG"``, ``"INFO"``).
                           Falls back to ``config/v1/logging.yaml`` → ``INFO``.
        strands_log_level: Level for ``strands.*`` loggers.
                           Falls back to ``WARNING``.
    """
    global _configured
    if _configured:
        return
    _configured = True

    os.makedirs(session_log_dir, exist_ok=True)

    cfg = _load_logging_yaml()
    resolved_level = log_level or str(cfg["level"])
    level = getattr(logging, resolved_level.upper(), logging.INFO)
    formatter = _JsonFormatter()

    # Root logger → app.jsonl (catches main, orchestrator, hooks, client, …)
    root = logging.getLogger()
    root.setLevel(level)
    _attach_file_handler(
        root,
        os.path.join(session_log_dir, DEFAULT_APP_LOG_FILENAME),
        level,
        formatter,
    )
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.WARNING)
    root.addHandler(stderr_handler)

    # Component loggers with propagate=False so they write only to their own
    # file and do NOT also appear in app.jsonl.
    seen_files: Dict[str, logging.FileHandler] = {}
    for namespace, filename in LOG_COMPONENT_FILES.items():
        comp_logger = logging.getLogger(namespace)
        comp_logger.setLevel(level)
        comp_logger.propagate = False
        if filename not in seen_files:
            fh = logging.FileHandler(
                os.path.join(session_log_dir, filename), encoding="utf-8"
            )
            fh.setFormatter(formatter)
            fh.setLevel(level)
            seen_files[filename] = fh
        comp_logger.addHandler(seen_files[filename])

    # strands.* loggers propagate to root (→ app.jsonl) but at a separate
    # level so verbose Strands internals don't flood the app log by default.
    # Set STRANDS_LOG_LEVEL=DEBUG in .env to see full Strands event loop traces.
    from src.constants import DEFAULT_STRANDS_LOG_LEVEL
    resolved_strands_level = strands_log_level or DEFAULT_STRANDS_LOG_LEVEL
    strands_level = getattr(logging, resolved_strands_level.upper(), logging.WARNING)
    logging.getLogger("strands").setLevel(strands_level)


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------


def get_logger(component: str) -> logging.Logger:
    """
    Return a named logger for *component*.

    Configuration (file handlers, level) is applied via
    ``init_session_logging()`` at startup.  If that function has not been
    called, Python's default handler (WARNING+ to stderr) is used — this is
    the expected behaviour in unit tests.

    Args:
        component: Logger name, typically ``__name__``.

    Returns:
        A ``logging.Logger`` instance.
    """
    return logging.getLogger(component)
