"""
ReMeLight-backed conversation manager for Strands Agent.

Replaces the default ``SlidingWindowConversationManager`` with a compaction
strategy: when ``agent.messages`` exceeds ``window_size``, the oldest batch
is summarised via ``MemoryProvider.compact()``, the summary is persisted as
episodic long-term memory via ``MemoryProvider.save()``, and the compacted
messages are replaced in-place with a single summary message.

When the LLM backend is a stub (returns empty summaries), the manager falls
back to a plain sliding-window drop so the agent never crashes from context
overflow.

This module lives under ``remelight/`` because it couples Strands'
``ConversationManager`` with ReMeLight's compactor — it will change when
the memory backend is swapped.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from strands.agent.conversation_manager import ConversationManager
from strands.types.exceptions import ContextWindowOverflowException

from src.memory.provider import MemoryProvider, Message

if TYPE_CHECKING:
    from strands.agent.agent import Agent

logger = logging.getLogger(__name__)

_SUMMARY_TOPIC = "compacted_dialog"


def _has_tool_result(message: dict) -> bool:
    """Return True if *message* contains at least one ``toolResult`` block."""
    return any("toolResult" in block for block in message.get("content", []))


def _has_tool_use(message: dict) -> bool:
    """Return True if *message* contains at least one ``toolUse`` block."""
    return any("toolUse" in block for block in message.get("content", []))


def _safe_split_index(messages: list[dict], raw_split: int) -> int:
    """Adjust *raw_split* so it does not orphan toolUse/toolResult pairs.

    Scans forward from *raw_split* until the boundary is safe:
      - The new first message is not a bare ``toolResult``.
      - A ``toolUse`` at the boundary is not separated from its ``toolResult``.

    Returns the adjusted index, or ``len(messages)`` if no safe boundary exists.
    """
    split = max(raw_split, 0)
    while split < len(messages):
        if _has_tool_result(messages[split]):
            split += 1
            continue
        if _has_tool_use(messages[split]):
            if split + 1 < len(messages) and not _has_tool_result(messages[split + 1]):
                split += 1
                continue
        break
    return split


def _strands_to_provider_messages(messages: list[dict]) -> list[Message]:
    """Convert Strands message dicts to ``MemoryProvider.Message`` dicts.

    Extracts role and flattens text content blocks into a single string.
    """
    result: list[Message] = []
    for msg in messages:
        texts: list[str] = []
        for block in msg.get("content", []):
            if isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
        result.append({
            "role": msg.get("role", "user"),
            "content": "\n".join(texts),
        })
    return result


class ReMeCompactionManager(ConversationManager):
    """Strands ``ConversationManager`` backed by ReMeLight compaction.

    After every agent invocation Strands calls ``apply_management``.  When the
    conversation exceeds ``window_size`` messages the oldest batch is compacted,
    persisted as episodic memory, and replaced with a single summary message.
    """

    def __init__(
        self,
        memory_provider: MemoryProvider,
        window_size: int = 40,
        compact_batch_size: int = 20,
        preserve_recent: int = 10,
    ) -> None:
        """Initialise the compaction manager.

        Args:
            memory_provider:   Provider used for ``compact()`` and ``save()``.
            window_size:       Message count threshold that triggers compaction.
            compact_batch_size: Number of oldest messages to compact per batch.
            preserve_recent:   Minimum messages to always keep (newest end).
        """
        super().__init__()
        self._provider = memory_provider
        self._window_size = window_size
        self._compact_batch_size = compact_batch_size
        self._preserve_recent = preserve_recent
        self._previous_summary: str = ""

    def apply_management(self, agent: "Agent", **kwargs: Any) -> None:
        """Compact oldest messages when the conversation exceeds the window.

        Called in Strands' ``finally`` block after every agent invocation.
        No-op when the message count is within ``window_size``.

        Args:
            agent: The agent whose ``messages`` list will be modified in-place.
        """
        messages = agent.messages
        if len(messages) <= self._window_size:
            return
        self._compact(messages)

    def flush(self, agent: "Agent") -> None:
        """Compact whatever remains in ``agent.messages`` at session end.

        Intended to be called once from the orchestrator's ``_close_session``
        before metrics are written, so any dialog that never crossed the
        per-turn window threshold is still persisted as episodic memory.

        No-op when the message list is empty.

        Args:
            agent: The agent whose ``messages`` list will be modified in-place.
        """
        messages = agent.messages
        if not messages:
            return
        self._do_compact_and_replace(messages, len(messages))

    def reduce_context(
        self,
        agent: "Agent",
        e: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """Aggressively reduce context on ``ContextWindowOverflowException``.

        Compacts all messages except the most recent ``preserve_recent``.
        Falls back to hard truncation if compaction still cannot reduce the
        context far enough.

        Args:
            agent: The agent whose ``messages`` list will be modified in-place.
            e:     The exception that triggered the reduction, if any.
        """
        messages = agent.messages
        if len(messages) <= self._preserve_recent:
            raise ContextWindowOverflowException(
                "Cannot reduce context further"
            ) from e

        split = max(len(messages) - self._preserve_recent, 1)
        split = _safe_split_index(messages, split)
        if split >= len(messages):
            raise ContextWindowOverflowException(
                "Cannot find safe split point"
            ) from e

        self._do_compact_and_replace(messages, split)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compact(self, messages: list[dict]) -> None:
        """Run one compaction cycle against *messages* (modified in-place)."""
        overflow = len(messages) - self._window_size
        split = min(overflow + self._compact_batch_size, len(messages) - self._preserve_recent)
        split = max(split, 1)
        split = _safe_split_index(messages, split)

        if split <= 0 or split >= len(messages):
            return

        self._do_compact_and_replace(messages, split)

    def _do_compact_and_replace(self, messages: list[dict], split: int) -> None:
        """Compact ``messages[:split]``, persist, and replace in-place."""
        to_compact = messages[:split]
        provider_msgs = _strands_to_provider_messages(to_compact)

        try:
            summary = self._provider.compact(provider_msgs)
        except Exception:
            logger.exception("compaction_manager: compact() raised — falling back to drop")
            summary = None

        if summary and summary.text:
            try:
                self._provider.save(
                    summary.text,
                    type="episodic",
                    topic=_SUMMARY_TOPIC,
                )
            except Exception:
                logger.exception("compaction_manager: save() raised for compacted summary")

            summary_message: dict = {
                "role": "user",
                "content": [{"text": f"[Previous conversation summary]\n{summary.text}"}],
            }
            messages[:] = [summary_message] + messages[split:]
            self.removed_message_count += split - 1
            self._previous_summary = summary.text
            logger.debug(
                "compaction_manager: compacted %d messages into summary",
                split,
                extra={"data": {"split": split, "summary_len": len(summary.text)}},
            )
        else:
            messages[:] = messages[split:]
            self.removed_message_count += split
            logger.warning(
                "compaction_manager: empty summary — dropped %d messages (stub LLM fallback)",
                split,
                extra={"data": {"split": split}},
            )
