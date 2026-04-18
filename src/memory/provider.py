"""
MemoryProvider protocol and supporting types.

The memory system exposes a single abstraction — ``MemoryProvider`` — so the
orchestrator and compaction manager can swap backing implementations (ReMeLight
now, Mem0 or another framework later) without changing call sites.

Session dialog is owned by Strands Agent (``agent.messages``); the provider
handles only long-term memory and context-window management via four methods:
``search``, ``save``, ``compact``, ``check_context``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Tuple, runtime_checkable

from typing_extensions import TypedDict


MemoryType = Literal["episodic", "semantic", "procedural"]


class Message(TypedDict, total=False):
    """One conversation message persisted to dialog history."""

    role: str           # "user" | "assistant" | "tool"
    content: str
    timestamp: str      # ISO-8601 UTC
    session_id: str


@dataclass(frozen=True)
class MemoryItem:
    """A single memory entry returned from search or produced by save."""

    key: str
    content: str
    type: MemoryType
    topic: Optional[str] = None
    timestamp: Optional[str] = None           # ISO-8601 UTC
    relevance_score: Optional[float] = None   # populated by search() only


@dataclass(frozen=True)
class SearchFilters:
    """Structured filter set for ``MemoryProvider.search``."""

    types: Optional[Tuple[MemoryType, ...]] = None
    topics: Optional[Tuple[str, ...]] = None
    date_range: Optional[Tuple[str, str]] = None   # (from_iso, to_iso)
    limit: int = 5


@dataclass(frozen=True)
class Summary:
    """Result of ``MemoryProvider.compact``."""

    text: str
    source_count: int


@runtime_checkable
class MemoryProvider(Protocol):
    """
    Provider-agnostic interface for the agent memory substrate.

    Implementations MUST be safe to call from the orchestrator's event loop and
    from hooks; they SHOULD serialise writes to shared state (JSONL appends,
    index updates) with a lock.
    """

    def search(self, query: str, filters: SearchFilters) -> list[MemoryItem]:
        """Return memories matching ``query`` filtered by ``filters``."""

    def save(
        self,
        content: str,
        type: MemoryType,
        topic: Optional[str] = None,
    ) -> MemoryItem:
        """Persist a new memory and return the stored item."""

    def compact(self, messages: list[Message]) -> Summary:
        """Condense ``messages`` into a summary suitable for long-term storage."""

    def check_context(
        self,
        messages: list[Message],
        budget_tokens: int,
    ) -> list[Message]:
        """Return a prefix of ``messages`` that fits within ``budget_tokens``."""
