"""
Search memory tool (V1.1 M3) — explicit-filter wrapper over ``MemoryProvider.search``.

Complements the per-turn auto-injection in ``Orchestrator._process_turn`` by
letting the agent issue a second, structured retrieval with its own filters
(type, topic, higher ``limit``) when the default context is insufficient.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, get_args

from strands import tool

from src.constants import VALID_MEMORY_TYPES
from src.logging_config import get_logger
from src.memory.provider import MemoryItem, MemoryProvider, MemoryType, SearchFilters
from src.tools.envelope import ToolResult, err, ok

logger = get_logger(__name__)

_HARD_LIMIT = 25


def _item_to_dict(item: MemoryItem) -> Dict[str, Any]:
    """Serialise a ``MemoryItem`` into a JSON-friendly dict for tool output."""
    return {
        "key": item.key,
        "content": item.content,
        "type": item.type,
        "topic": item.topic,
        "timestamp": item.timestamp,
        "relevance_score": item.relevance_score,
    }


def create_search_memory_tool(memory_provider: MemoryProvider):
    """
    Create a Strands @tool that queries long-term memory with explicit filters.

    Args:
        memory_provider: The MemoryProvider instance to read from.

    Returns:
        A Strands tool function that returns a ``ToolResult`` envelope whose
        ``data`` is a list of serialised memory items.
    """

    @tool
    def search_memory(
        query: str,
        type: Optional[Literal["episodic", "semantic", "procedural"]] = None,
        topic: Optional[str] = None,
        limit: int = 10,
    ) -> ToolResult:
        """
        Search long-term memory for items matching ``query`` and the given filters.

        Args:
            query: Free-text query; case-insensitive substring match on stored content.
            type:  Optional memory-type filter (episodic/semantic/procedural).
            topic: Optional topic-tag filter.
            limit: Maximum number of items to return (1..25).

        Returns:
            ``ToolResult`` envelope.  ``data`` is a list of matching memory dicts
            (most relevant first); ``metadata`` carries the effective filters.
        """
        clamped = max(1, min(int(limit), _HARD_LIMIT))

        normalised_type: Optional[MemoryType] = None
        if type is not None:
            folded = type.casefold()
            if folded not in VALID_MEMORY_TYPES:
                return err(
                    f"type must be one of {sorted(VALID_MEMORY_TYPES)}",
                    field="type",
                )
            normalised_type = folded  # type: ignore[assignment]

        normalised_topic = topic.strip() if isinstance(topic, str) else None
        if normalised_topic == "":
            normalised_topic = None

        filters = SearchFilters(
            types=(normalised_type,) if normalised_type else None,
            topics=(normalised_topic,) if normalised_topic else None,
            limit=clamped,
        )

        try:
            matches: List[MemoryItem] = memory_provider.search(query or "", filters)
        except Exception as exc:
            logger.error(
                "search_memory: provider search failed",
                extra={"data": {"error": str(exc)}},
            )
            return err(f"search failed: {exc}")

        logger.debug(
            "search_memory: completed",
            extra={"data": {
                "query_len": len(query or ""),
                "type": normalised_type,
                "topic": normalised_topic,
                "limit": clamped,
                "result_count": len(matches),
            }},
        )

        return ok(
            [_item_to_dict(m) for m in matches],
            type=normalised_type,
            topic=normalised_topic,
            limit=clamped,
            result_count=len(matches),
        )

    return search_memory
