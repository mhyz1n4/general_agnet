"""
Per-session tool-result cache (V1.1 M7).

Scope
-----
A small LRU keyed by ``(tool_name, args_hash)`` that lives for the duration of
one chat session.  The Orchestrator constructs one ``SessionToolCache`` per
session and binds it to a ContextVar before each turn so that tools wrapped
with ``@cached_tool`` can look it up without the Orchestrator needing to pass
the cache through every tool signature.

Out of scope (per plan): cross-session cache, TTL, persistence, fuzzy/embed
cache.  Those live with V1.2/V2.

Thread-safety
-------------
The cache is single-threaded per session (the Orchestrator invokes the agent
on a single worker thread).  No locking is needed; the ContextVar copy is
propagated into the worker via ``contextvars.copy_context()`` in the
Orchestrator.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
from collections import OrderedDict
from typing import Any, Callable, Optional, TypeVar

from src.logging_config import get_logger
from src.tools.envelope import ToolResult

logger = get_logger(__name__)

DEFAULT_MAX_ENTRIES = 64


class SessionToolCache:
    """
    LRU-bounded cache of tool results scoped to one chat session.

    Entries are keyed by ``(tool_name, args_hash)`` where ``args_hash`` is a
    deterministic hash of the tool's call arguments.  Hitting an existing entry
    moves it to the most-recently-used end; evictions drop the oldest key when
    the cache is full.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        """
        Initialise an empty cache.

        Args:
            max_entries: Maximum number of entries retained before LRU eviction.
        """
        self._entries: "OrderedDict[tuple[str, str], ToolResult]" = OrderedDict()
        self._max_entries = max_entries
        self.hits: int = 0
        self.misses: int = 0

    def get(self, tool_name: str, args_hash: str) -> Optional[ToolResult]:
        """
        Return the cached envelope for ``(tool_name, args_hash)`` or ``None``.

        A hit promotes the entry to most-recently-used and increments ``hits``.
        A miss increments ``misses``.
        """
        key = (tool_name, args_hash)
        if key in self._entries:
            self._entries.move_to_end(key)
            self.hits += 1
            return self._entries[key]
        self.misses += 1
        return None

    def set(self, tool_name: str, args_hash: str, value: ToolResult) -> None:
        """
        Store ``value`` under ``(tool_name, args_hash)``, evicting LRU if full.
        """
        key = (tool_name, args_hash)
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = value
        if len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        """Return the number of entries currently held."""
        return len(self._entries)


_session_cache: contextvars.ContextVar[Optional[SessionToolCache]] = contextvars.ContextVar(
    "session_tool_cache", default=None
)


def set_session_cache(cache: Optional[SessionToolCache]) -> contextvars.Token:
    """Bind ``cache`` to the current context; returns the reset token."""
    return _session_cache.set(cache)


def reset_session_cache(token: contextvars.Token) -> None:
    """Undo a previous ``set_session_cache``."""
    _session_cache.reset(token)


def get_session_cache() -> Optional[SessionToolCache]:
    """Return the cache bound to the current context, if any."""
    return _session_cache.get()


def _hash_args(args: tuple, kwargs: dict) -> str:
    """
    Return a deterministic short hash for ``(args, kwargs)``.

    Arguments that are not JSON-serialisable fall back to ``repr`` so the hash
    is still stable; non-cacheable tools should simply not use ``@cached_tool``.
    """
    try:
        payload = json.dumps(
            {"args": list(args), "kwargs": dict(sorted(kwargs.items()))},
            sort_keys=True,
            default=repr,
        )
    except (TypeError, ValueError):
        payload = repr((args, sorted(kwargs.items())))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


F = TypeVar("F", bound=Callable[..., ToolResult])


def cached_tool(tool_name: str) -> Callable[[F], F]:
    """
    Decorator: cache a tool's ``ToolResult`` in the ambient ``SessionToolCache``.

    Only successful envelopes (``ok=True``) are cached — we do not want a
    transient network failure to lock in as the answer for the remainder of
    the session.  If no cache is bound to the context the wrapped function is
    invoked directly with no caching side-effects.

    Args:
        tool_name: Stable name used as the first component of the cache key.
                   Must be unique per logical tool.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> ToolResult:
            cache = get_session_cache()
            if cache is None:
                return fn(*args, **kwargs)

            key = _hash_args(args, kwargs)
            cached = cache.get(tool_name, key)
            if cached is not None:
                logger.debug(
                    "cached_tool: hit",
                    extra={"data": {"tool": tool_name, "args_hash": key}},
                )
                return cached

            result = fn(*args, **kwargs)
            if isinstance(result, dict) and result.get("ok"):
                cache.set(tool_name, key, result)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
