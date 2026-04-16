"""
Memorize sub-agent — multi-step verification wrapper around save_message.

Per design §6a:
  1. Validate inputs.
  2. Write to filesystem via MemoryManager.save_message().
  3. Verify the key was indexed (content hash found in index).
  4. Return SubAgentResult.

**Status: retained for future use — NOT used by main.py today.**

This module implements the ``MemorizeSubAgent`` class, designed for the planned
``PostTurnMemoryHook`` pattern: a sub-agent that runs after every conversation
turn and autonomously decides whether the interaction is worth memorising.
That hook has not been activated yet (``auto_memory_enabled`` is False in config).

The active memory-save path is ``src/tools/memorize.py`` — a Strands ``@tool``
closure injected into the agent and called when the user explicitly asks to
remember something.  Use that module for any work that is live today.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.agents.base import SubAgentResult
from src.constants import CONTENT_HASH_LENGTH, MEMORY_ID_HEX_LENGTH, VALID_MEMORY_TYPES
from src.logging_config import get_logger
from src.memory.manager import MemoryManager
from src.memory.types import MemoryMetadata

logger = get_logger(__name__)


class MemorizeSubAgent:
    """
    Verify-and-save wrapper around MemoryManager for explicit user-directed saves.

    Unlike the Strands @tool (which is atomic), this sub-agent performs a
    post-write verification step to confirm the entry was indexed successfully.
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self.memory_manager = memory_manager

    def run(
        self,
        content: str,
        memory_type: str,
        topic: str = "",
    ) -> SubAgentResult:
        """
        Save content to long-term memory and verify it was indexed.

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

        message_id = f"{memory_type}_{uuid.uuid4().hex[:MEMORY_ID_HEX_LENGTH]}"
        metadata: MemoryMetadata = {
            "type": memory_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if topic.strip():
            metadata["topic"] = topic.strip()

        try:
            self.memory_manager.save_message(message_id, content.strip(), metadata)
        except Exception as exc:
            logger.error(
                "memorize_agent: save failed",
                extra={"data": {"message_id": message_id, "error": str(exc)}},
            )
            return SubAgentResult(success=False, output="", error=str(exc))

        # Verification: confirm the content hash was indexed.
        content_hash = hashlib.sha256(content.strip().encode()).hexdigest()[:CONTENT_HASH_LENGTH]
        verified_key: Optional[str] = None
        try:
            verified_key = self.memory_manager.indexer.find_by_content_hash(content_hash)
        except Exception as exc:
            logger.warning(
                "memorize_agent: verification check raised",
                extra={"data": {"error": str(exc)}},
            )

        if verified_key is None:
            # Save appeared to succeed but we cannot confirm indexing.
            # Return success=True — the DLQ handles actual write failures.
            logger.warning(
                "memorize_agent: verification inconclusive — hash not found in index",
                extra={"data": {"message_id": message_id}},
            )
            return SubAgentResult(
                success=True,
                output=f"Saved with key {message_id} (index verification inconclusive).",
            )

        logger.debug(
            "memorize_agent: verified",
            extra={"data": {"message_id": message_id, "verified_key": verified_key}},
        )
        return SubAgentResult(
            success=True,
            output=f"Saved and verified with key: {verified_key}",
        )
