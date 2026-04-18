"""
Memorize sub-agent — multi-step wrapper around MemoryProvider.save().

Per design SS6a (V1.1):
  1. Validate inputs.
  2. Write to memory via MemoryProvider.save().
  3. Return SubAgentResult.

Post-write verification (content hash check) has been retired — the provider
handles indexing internally.
"""

import uuid
from typing import Optional

from src.agents.base import SubAgentResult
from src.constants import MEMORY_ID_HEX_LENGTH, VALID_MEMORY_TYPES
from src.logging_config import get_logger
from src.memory.provider import MemoryProvider

logger = get_logger(__name__)


class MemorizeSubAgent:
    """
    Verify-and-save wrapper around MemoryProvider for explicit user-directed saves.
    """

    def __init__(self, memory_provider: MemoryProvider) -> None:
        """
        Initialise the memorize sub-agent.

        Args:
            memory_provider: ``MemoryProvider`` implementation to save through.
        """
        self.memory_provider = memory_provider

    def run(
        self,
        content: str,
        memory_type: str,
        topic: str = "",
    ) -> SubAgentResult:
        """
        Save content to long-term memory via the provider.

        Args:
            content:     Text to persist.
            memory_type: One of "episodic", "semantic", "procedural".
            topic:       Optional category label (e.g. "work", "personal").

        Returns:
            SubAgentResult with success=True and the assigned key, or
            success=False with an error description.
        """
        logger.debug(
            "memorize_agent: starting",
            extra={"data": {"type": memory_type, "topic": topic or None}},
        )

        if not content.strip():
            return SubAgentResult(success=False, output="", error="Content cannot be empty.")
        if memory_type not in VALID_MEMORY_TYPES:
            return SubAgentResult(
                success=False,
                output="",
                error=f"type must be one of {sorted(VALID_MEMORY_TYPES)}, got '{memory_type}'.",
            )

        try:
            item = self.memory_provider.save(
                content.strip(),
                type=memory_type,
                topic=topic.strip() or None,
            )
        except Exception as exc:
            logger.error(
                "memorize_agent: save failed",
                extra={"data": {"error": str(exc)}},
            )
            return SubAgentResult(success=False, output="", error=str(exc))

        logger.debug(
            "memorize_agent: saved",
            extra={"data": {"key": item.key, "type": memory_type}},
        )
        return SubAgentResult(
            success=True,
            output=f"Saved with key: {item.key}",
        )
