"""
Memorize tool — Strands @tool closure for saving information to long-term memory.

Usage:
    from src.tools.memorize import create_memorize_tool
    memorize_tool = create_memorize_tool(memory_manager)
    agent = Agent(model=..., tools=[memorize_tool], system_prompt=...)
"""

from datetime import datetime, timezone
from uuid import uuid4

from strands import tool

from src.logging_config import get_logger
from src.memory.manager import MemoryManager

logger = get_logger(__name__)

VALID_TYPES = {"episodic", "semantic", "procedural"}


def create_memorize_tool(memory_manager: MemoryManager):
    """
    Create a Strands @tool that saves content to long-term memory.

    Args:
        memory_manager: The MemoryManager instance to write through.

    Returns:
        A Strands tool function decorated with @tool.
    """

    @tool
    def memorize(content: str, type: str, topic: str = "") -> str:
        """
        Save information to long-term memory.

        Args:
            content: The information to remember.
            type: Memory type — one of 'episodic', 'semantic', or 'procedural'.
            topic: Optional category label (e.g. 'work', 'preferences').

        Returns:
            Confirmation string with the assigned memory key, or an error message.
        """
        logger.debug(
            "memorize: called",
            extra={"data": {"type": type, "topic": topic, "content_len": len(content)}},
        )

        if not content.strip():
            logger.warning("memorize: rejected — empty content")
            return "Error: content cannot be empty."

        if type not in VALID_TYPES:
            logger.warning(
                "memorize: rejected — invalid type",
                extra={"data": {"type": type}},
            )
            return f"Error: type must be one of {sorted(VALID_TYPES)}."

        message_id = f"{type}_{uuid4().hex[:8]}"
        metadata = {
            "type": type,
            "topic": topic.strip() or None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            memory_manager.save_message(message_id, content.strip(), metadata)
            logger.debug(
                "memorize: saved",
                extra={"data": {"message_id": message_id, "type": type}},
            )
            return f"Saved to memory with key: {message_id}"
        except Exception as exc:
            logger.error(
                "memorize: save failed",
                extra={"data": {"message_id": message_id, "error": str(exc)}},
            )
            return f"Error saving to memory: {str(exc)}"

    return memorize
