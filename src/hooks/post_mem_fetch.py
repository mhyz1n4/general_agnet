"""
Post-memory-fetch hook — truncates oversized context and saves a retrieval_event
to the session memory for session-scoped recall.
"""

from typing import List, Optional

from src.constants import CONTEXT_BLOCK_DELIMITER, DEFAULT_MAX_CONTEXT_CHARS, MEMORY_ID_HEX_LENGTH, QUERY_LOG_PREVIEW_LENGTH
from src.logging_config import get_logger
from src.memory.base import BaseStorage
from .base import BaseHook, HookResult

logger = get_logger(__name__)


class PostMemFetchHook(BaseHook):
    """Truncate context to max_context_chars and save a retrieval_event."""

    def __init__(
        self,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        session_storage: Optional[BaseStorage] = None,
    ) -> None:
        self.max_context_chars = max_context_chars
        self.session_storage = session_storage

    def run(
        self,
        context: str,
        retrieved_keys: Optional[List[str]] = None,
        query: str = "",
        **kwargs: object,
    ) -> HookResult:
        """
        Truncate context to the configured character limit and log a retrieval event.

        The truncated context is returned in ``HookResult.message``.  A
        ``retrieval_event`` record is written to ``session_storage`` (if configured)
        so the current session can recall which keys were retrieved for this query.

        Args:
            context:        Formatted memory context string from the retriever.
            retrieved_keys: List of memory keys that contributed to *context*.
            query:          Original user query (used to annotate the retrieval event).
            **kwargs:       Unused; present for ``BaseHook`` compatibility.

        Returns:
            ``HookResult(success=True, message=truncated_context)``.
        """
        retrieved_keys = retrieved_keys or []

        logger.debug(
            "post_mem_fetch: checking context size",
            extra={"data": {"chars": len(context), "limit": self.max_context_chars}},
        )

        truncated_context = self._truncate(context)

        if self.session_storage is not None:
            import uuid
            from datetime import datetime, timezone

            event_id = f"retrieval_{uuid.uuid4().hex[:MEMORY_ID_HEX_LENGTH]}"
            event = {
                "id": event_id,
                "content": f"Retrieved keys for query '{query[:QUERY_LOG_PREVIEW_LENGTH]}': {retrieved_keys}",
                "metadata": {
                    "type": "retrieval_event",
                    "query": query[:200],
                    "keys": retrieved_keys,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            try:
                self.session_storage.save(event_id, event)
            except Exception as exc:
                logger.warning(
                    "post_mem_fetch: failed to save retrieval_event",
                    extra={"data": {"error": str(exc)}},
                )

        return HookResult(success=True, message=truncated_context)

    def _truncate(self, context: str) -> str:
        """
        Trim *context* to at most ``max_context_chars`` characters.

        Splits on ``CONTEXT_BLOCK_DELIMITER`` boundaries and greedily includes blocks
        until the budget is exhausted.  If a single block already exceeds the
        limit, it is returned as-is (no mid-block truncation).  Returns the
        original string unchanged when it is within the limit.

        Args:
            context: Formatted context string, possibly multi-block.

        Returns:
            A string whose length is ``<= max_context_chars``, or the original
            string if it is within the limit or consists of a single oversized block.
        """
        if not context or len(context) <= self.max_context_chars:
            return context

        # Split on block boundaries and keep top blocks that fit
        blocks = context.split(CONTEXT_BLOCK_DELIMITER)
        header = ""
        body_blocks = []

        if blocks and not blocks[0].startswith(" "):
            # first split element is the text before the first delimiter (usually "")
            header = blocks[0]
            body_blocks = blocks[1:]
        else:
            body_blocks = blocks

        kept: List[str] = []
        total = len(header)

        for block in body_blocks:
            reconstructed = CONTEXT_BLOCK_DELIMITER + block
            if total + len(reconstructed) > self.max_context_chars:
                break
            kept.append(reconstructed)
            total += len(reconstructed)

        if not kept and body_blocks:
            # Single block already exceeds limit — return as-is
            return context

        result = header + "".join(kept)
        logger.debug(
            "post_mem_fetch: truncated context",
            extra={"data": {"original_chars": len(context), "truncated_chars": len(result)}},
        )
        return result
