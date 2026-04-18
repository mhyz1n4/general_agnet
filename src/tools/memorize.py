"""
Memorize tool — Strands @tool closure for saving information to long-term memory.

V1.1: Uses ``MemoryProvider.save()`` instead of ``MemoryManager.save_message()``.
The tool is created once per session via ``create_memorize_tool(memory_provider)``
and injected into the Strands ``Agent`` as a callable tool.
"""

from typing import Literal, Optional

from strands import tool

from src.constants import VALID_MEMORY_TYPES
from src.logging_config import get_logger
from src.memory.provider import MemoryProvider

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
    ) -> str:
        """
        Save information to long-term memory. Call this only when the user
        explicitly asks to remember, save, or note something.

        Args:
            content: The information to remember. Must be non-empty.
            type: Memory category — episodic (events/conversations), semantic (facts/preferences), procedural (how-tos/workflows).
            topic: Optional category label (e.g. 'work', 'preferences').

        Returns:
            Confirmation string or an error message.
        """
        logger.debug(
            "memorize: called",
            extra={"data": {"type": type, "topic": topic, "content_len": len(content)}},
        )

        if not content.strip():
            logger.warning("memorize: rejected — empty content")
            return "Error: content cannot be empty."

        type = type.casefold()

        if type not in VALID_MEMORY_TYPES:
            logger.warning(
                "memorize: rejected — invalid type",
                extra={"data": {"type": type}},
            )
            return f"Error: type must be one of {sorted(VALID_MEMORY_TYPES)}."

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
            return f"Saved to memory with key: {item.key}"
        except Exception as exc:
            logger.error(
                "memorize: save failed",
                extra={"data": {"error": str(exc)}},
            )
            return f"Error saving to memory: {str(exc)}"

    return memorize
