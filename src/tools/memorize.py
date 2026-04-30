"""
Memorize tool — Strands @tool closure for saving information to long-term memory.

V1.1: Uses ``MemoryProvider.save()`` instead of ``MemoryManager.save_message()``
and returns the common ``ToolResult`` envelope so downstream code can read
success / error uniformly across tools.
"""

from typing import Literal, Optional

from strands import tool

from src.constants import VALID_MEMORY_TYPES
from src.logging_config import get_logger
from src.memory.provider import MemoryProvider
from src.tools.envelope import ToolResult, err, ok

logger = get_logger(__name__)


def create_memorize_tool(memory_provider: MemoryProvider, session_id: Optional[str] = None):
    """
    Create a Strands @tool that saves content to long-term memory.

    Args:
        memory_provider: The MemoryProvider instance to write through.
        session_id: Current session ID (used as topic for episodic entries).

    Returns:
        A Strands tool function decorated with @tool.
    """

    @tool
    def memorize(
        content: str,
        type: Literal["episodic", "semantic", "procedural"],
        topic: str = "",
    ) -> ToolResult:
        """
        Save information to long-term memory. Call this only when the user
        explicitly asks to remember, save, or note something.

        Args:
            content: The information to remember. Must be non-empty.
            type: Memory category — episodic (events/conversations), semantic (facts/preferences), procedural (how-tos/workflows).
            topic: Optional category label (e.g. 'work', 'preferences').

        Returns:
            ``ToolResult`` envelope. On success ``data`` carries the saved key;
            on failure ``error`` carries a human-readable message.
        """
        logger.debug(
            "memorize: called",
            extra={"data": {"type": type, "topic": topic, "content_len": len(content)}},
        )

        if not content.strip():
            logger.warning("memorize: rejected — empty content")
            return err("content cannot be empty")

        type = type.casefold()

        if type not in VALID_MEMORY_TYPES:
            logger.warning(
                "memorize: rejected — invalid type",
                extra={"data": {"type": type}},
            )
            return err(f"type must be one of {sorted(VALID_MEMORY_TYPES)}")

        resolved_topic = topic.strip() or None
        if type == "episodic" and session_id is not None and not resolved_topic:
            resolved_topic = session_id

        try:
            item = memory_provider.save(
                content.strip(),
                type=type,
                topic=resolved_topic,
            )
            logger.debug(
                "memorize: saved",
                extra={"data": {"key": item.key, "type": type}},
            )
            return ok({"key": item.key}, type=type, topic=resolved_topic)
        except Exception as exc:
            logger.error(
                "memorize: save failed",
                extra={"data": {"error": str(exc)}},
            )
            return err(f"save failed: {exc}")

    return memorize
