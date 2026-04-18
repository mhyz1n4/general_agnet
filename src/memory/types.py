"""
Shared TypedDicts for the memory system.

V1.1: Retains ``TurnRecord`` and ``SessionMetricsDict`` used by the
orchestrator and post-session hook.  V1-only types (``MemoryMetadata``,
``StorageRecord``, ``IndexEntry``) have been retired.
"""

from __future__ import annotations

from typing import List, Optional
from typing_extensions import TypedDict


class TurnRecord(TypedDict):
    """One conversation turn stored in SessionMetrics."""
    user: str
    assistant: str


class SessionMetricsDict(TypedDict, total=False):
    """
    Serialised form of SessionMetrics passed to PostSessionHook.

    All fields are optional (total=False) because the dict is constructed
    by ``SessionMetrics.to_dict()`` and callers should not assume every
    field is always present.
    """
    duration_seconds: float
    turn_count: int
    memory_hits: int
    memory_misses: int
    llm_token_count_est: int
    llm_latency_ms: float
    tool_calls_made: int
    tool_failures: int
    storage_size_bytes: int
    turns: List[TurnRecord]


class LoggingConfig(TypedDict, total=False):
    """Parsed content of config/v1/logging.yaml."""
    path: str
    level: str
    max_dir_bytes: int
