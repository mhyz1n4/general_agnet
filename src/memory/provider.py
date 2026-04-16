"""
MemoryProvider protocol and supporting types.

The V1.1 memory system exposes a single abstraction — ``MemoryProvider`` — so
the orchestrator can swap backing implementations (ReMeLight now, Mem0 or
another framework later) without changing any call sites.

All cross-provider data flows through the dataclasses defined here. Provider
implementations translate between these and their native representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class ReasoningContext:
    """Input/output of ``MemoryProvider.pre_reasoning_hook``."""

    messages: list[Message] = field(default_factory=list)
    memory_items: list[MemoryItem] = field(default_factory=list)
    budget_tokens: int = 0


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

    def save_session_turn(self, message: Message) -> None:
        """Append a single turn to the active session's dialog log."""

    def get_session_history(self, session_id: str) -> list[Message]:
        """Load the full dialog history for ``session_id``."""

    def pre_reasoning_hook(self, context: ReasoningContext) -> ReasoningContext:
        """
        Adjust ``context`` before the LLM call.

        Typically chains tool-result compaction, context budget enforcement,
        and async summarisation. Providers MAY return ``context`` unchanged.
        """
