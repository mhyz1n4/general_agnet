"""
Post-memory-fetch hook — truncates oversized context and saves a retrieval_event
to the session memory for session-scoped recall.
"""

from typing import Any, List, Optional

from src.logging_config import get_logger
from .base import BaseHook, HookResult

logger = get_logger(__name__)

_BLOCK_DELIMITER = "--- Context (Key:"


class PostMemFetchHook(BaseHook):
    """Truncate context to max_context_chars and save a retrieval_event."""

    def __init__(
        self,
        max_context_chars: int = 8000,
        session_storage: Optional[Any] = None,
    ) -> None:
        self.max_context_chars = max_context_chars
        self.session_storage = session_storage

    def run(
        self,
        context: str,
        retrieved_keys: Optional[List[str]] = None,
        query: str = "",
        **kwargs: Any,
    ) -> HookResult:
        retrieved_keys = retrieved_keys or []

        logger.debug(
            "post_mem_fetch: checking context size",
            extra={"data": {"chars": len(context), "limit": self.max_context_chars}},
        )

        truncated_context = self._truncate(context)

        if self.session_storage is not None:
            import uuid
            from datetime import datetime, timezone

            event_id = f"retrieval_{uuid.uuid4().hex[:8]}"
            event = {
                "id": event_id,
                "content": f"Retrieved keys for query '{query[:80]}': {retrieved_keys}",
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
        if not context or len(context) <= self.max_context_chars:
            return context

        # Split on block boundaries and keep top blocks that fit
        blocks = context.split(_BLOCK_DELIMITER)
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
            reconstructed = _BLOCK_DELIMITER + block
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
