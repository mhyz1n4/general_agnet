"""
Shared TypedDicts for the memory system.

These types describe the concrete data structures that flow between
storage, indexing, retrieval, and the orchestrator. Importing from this
module eliminates Dict[str, Any] throughout the codebase.
"""

from __future__ import annotations

from typing import List, Optional
from typing_extensions import TypedDict


class MemoryMetadata(TypedDict, total=False):
    """
    Metadata attached to every stored memory item.

    All fields are optional (total=False) because different memory types
    carry different subsets (e.g. episodic items have session_id/turn,
    semantic items have topic, retrieval events have query/keys).
    """
    type: str               # "episodic" | "semantic" | "procedural"
    topic: Optional[str]
    timestamp: str          # ISO-8601 UTC
    content_hash: str       # 16-char SHA-256 prefix for deduplication
    session_id: str
    turn: int
    # Retrieval-event fields (written by PostMemFetchHook)
    query: str
    keys: List[str]


class StorageRecord(TypedDict):
    """The envelope written to and read from any BaseStorage backend."""
    id: str
    content: str
    metadata: MemoryMetadata


class IndexEntry(TypedDict):
    """One entry in the JSON index managed by JSONIndexer."""
    metadata: MemoryMetadata
    keywords: List[str]


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
    # LLM usage (estimated from character counts; 1 token ≈ 4 chars — not from API)
    llm_token_count_est: int
    llm_latency_ms: float
    # Tool tracking
    tool_calls_made: int
    tool_failures: int
    # Storage
    eviction_count: int
    index_size_bytes: int
    storage_size_bytes: int
    turns: List[TurnRecord]


class LoggingConfig(TypedDict, total=False):
    """Parsed content of config/v1/logging.yaml."""
    path: str
    level: str
    max_dir_bytes: int
