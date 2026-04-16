"""
In-memory ``MemoryProvider`` implementation for tests.

``StubMemoryProvider`` keeps everything in RAM: a ``dict[str, MemoryItem]`` for
saved memories and a ``list[Message]`` per session for dialog history. It has
no persistence, no indexing, no LLM dependency — substring matching stands in
for retrieval. Tests use it to exercise the orchestrator against a deterministic
provider without pulling in ReMeLight.
"""

from __future__ import annotations

import uuid
from typing import Optional

from src.constants import CHARS_PER_TOKEN

from .provider import (
    MemoryItem,
    MemoryProvider,
    MemoryType,
    Message,
    ReasoningContext,
    SearchFilters,
    Summary,
)


class StubMemoryProvider:
    """Deterministic in-memory provider for unit and integration tests."""

    def __init__(self) -> None:
        """Initialise empty in-memory stores for items and session histories."""
        self._items: dict[str, MemoryItem] = {}
        self._sessions: dict[str, list[Message]] = {}

    def search(self, query: str, filters: SearchFilters) -> list[MemoryItem]:
        """
        Return memories whose content contains ``query`` (case-insensitive) and
        match every non-``None`` field in ``filters``.

        Args:
            query:   Substring to search for; empty string matches all items
                     that pass the filter predicates.
            filters: Structured filter set. ``types`` and ``topics`` are applied
                     as whitelist filters; ``limit`` caps the result length.

        Returns:
            Matching ``MemoryItem`` list, sorted by timestamp descending
            (absent timestamps sort to the end), capped at ``filters.limit``.
        """
        query_lower = query.lower()
        matches: list[MemoryItem] = []
        for item in self._items.values():
            if filters.types is not None and item.type not in filters.types:
                continue
            if filters.topics is not None and item.topic not in filters.topics:
                continue
            if query_lower and query_lower not in item.content.lower():
                continue
            matches.append(item)
        matches.sort(key=lambda it: it.timestamp or "", reverse=True)
        return matches[: filters.limit]

    def save(
        self,
        content: str,
        type: MemoryType,
        topic: Optional[str] = None,
    ) -> MemoryItem:
        """
        Persist a memory item in RAM and return it.

        Args:
            content: The text to store.
            type:    Memory type literal (episodic/semantic/procedural).
            topic:   Optional category label.

        Returns:
            The stored ``MemoryItem`` with a generated 12-char hex key and
            ``timestamp=None``.
        """
        key = uuid.uuid4().hex[:12]
        item = MemoryItem(
            key=key,
            content=content,
            type=type,
            topic=topic,
            timestamp=None,
        )
        self._items[key] = item
        return item

    def compact(self, messages: list[Message]) -> Summary:
        """
        Produce a deterministic ``role: content`` concatenation — stand-in for
        a real LLM summarisation so tests don't depend on a provider.

        Args:
            messages: Messages to summarise; order is preserved in the output.

        Returns:
            ``Summary`` whose ``text`` is the joined lines and
            ``source_count`` is ``len(messages)``.
        """
        text = "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}" for m in messages
        )
        return Summary(text=text, source_count=len(messages))

    def check_context(
        self,
        messages: list[Message],
        budget_tokens: int,
    ) -> list[Message]:
        """
        Keep the most recent messages that fit within ``budget_tokens``.

        Token cost is approximated as ``len(content) / CHARS_PER_TOKEN``.
        Iteration is newest-first; the final list preserves original order.

        Args:
            messages:      Full conversation history (oldest-first).
            budget_tokens: Maximum tokens allowed in the returned slice.

        Returns:
            A suffix of ``messages`` whose cumulative char size is within
            ``budget_tokens * CHARS_PER_TOKEN``.
        """
        budget_chars = max(0, budget_tokens * CHARS_PER_TOKEN)
        kept: list[Message] = []
        used = 0
        for m in reversed(messages):
            size = len(m.get("content", ""))
            if used + size > budget_chars:
                break
            kept.append(m)
            used += size
        kept.reverse()
        return kept

    def save_session_turn(self, message: Message) -> None:
        """
        Append ``message`` to the in-memory history for its session.

        Args:
            message: Message dict; its ``session_id`` key (empty string when
                     missing) selects the target history list.
        """
        session_id = message.get("session_id", "")
        self._sessions.setdefault(session_id, []).append(message)

    def get_session_history(self, session_id: str) -> list[Message]:
        """
        Return a copy of the message list for ``session_id``.

        Args:
            session_id: Session identifier to look up.

        Returns:
            A new list (safe for caller mutation) of stored messages, or an
            empty list when the session is unknown.
        """
        return list(self._sessions.get(session_id, []))

    def pre_reasoning_hook(self, context: ReasoningContext) -> ReasoningContext:
        """
        Pass ``context`` through unchanged — the stub applies no pre-reasoning
        transformations.

        Args:
            context: Input reasoning context.

        Returns:
            The same ``ReasoningContext`` instance.
        """
        return context


# Runtime protocol check: keep the module self-verifying.
assert isinstance(StubMemoryProvider(), MemoryProvider)
