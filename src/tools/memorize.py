"""
Memorize tool — Strands @tool closure for saving information to long-term memory.

**Status: active — this is the memory-save path used by main.py.**

The tool is created once per session via ``create_memorize_tool(memory_manager)``
and injected into the Strands ``Agent`` as a callable tool.  The LLM calls it
when the user explicitly asks to remember, save, or note something.  The allowed
memory types are enforced via a ``Literal`` type annotation so the Strands framework
includes them in the JSON schema sent to the model, providing sampler-level
validation without relying on system-prompt rules.

See ``src/agents/memorize.py`` for the ``MemorizeSubAgent`` class — a retained
sub-agent designed for the future ``PostTurnMemoryHook`` auto-memory pattern
(not active today).

Usage:
    from src.tools.memorize import create_memorize_tool
    memorize_tool = create_memorize_tool(memory_manager)
    agent = Agent(model=..., tools=[memorize_tool], system_prompt=...)
"""

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from strands import tool

from src.constants import MEMORY_ID_HEX_LENGTH, MEMORY_TIMESTAMP_DISPLAY_FORMAT, VALID_MEMORY_TYPES
from src.logging_config import get_logger
from src.memory.manager import MemoryManager

logger = get_logger(__name__)


def create_memorize_tool(memory_manager: MemoryManager, session_id: Optional[str] = None):
    """
    Create a Strands @tool that saves content to long-term memory.

    Args:
        memory_manager: The MemoryManager instance to write through.
        session_id: Current session ID. When provided, episodic entries are
            stored under conversations/{session_id}/ instead of unknown_session/.

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

        # Normalize type casing silently (e.g. "Episodic" → "episodic")
        type = type.casefold()

        if type not in VALID_MEMORY_TYPES:
            logger.warning(
                "memorize: rejected — invalid type",
                extra={"data": {"type": type}},
            )
            return f"Error: type must be one of {sorted(VALID_MEMORY_TYPES)}."

        now = datetime.now(timezone.utc)
        timestamp_iso = now.isoformat()
        enriched_content = f"[{now.strftime(MEMORY_TIMESTAMP_DISPLAY_FORMAT)}] {content.strip()}"

        message_id = f"{type}_{uuid4().hex[:MEMORY_ID_HEX_LENGTH]}"
        metadata: dict = {
            "type": type,
            "topic": topic.strip() or None,
            "timestamp": timestamp_iso,
        }
        # Episodic entries are scoped to the current session so they land in the
        # right folder (conversations/{session_id}/…) rather than unknown_session/.
        if type == "episodic" and session_id is not None:
            metadata["session_id"] = session_id

        try:
            memory_manager.save_message(message_id, enriched_content, metadata)
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
