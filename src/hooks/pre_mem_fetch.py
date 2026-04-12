"""
Pre-memory-fetch hook — normalises the user query before memory retrieval.

Normalisation: strip non-alphanumeric characters (except spaces and apostrophes),
lowercase, collapse whitespace.
"""

import re

from src.constants import QUERY_LOG_PREVIEW_LENGTH
from src.logging_config import get_logger
from .base import BaseHook, HookResult

logger = get_logger(__name__)


class PreMemFetchHook(BaseHook):
    """Normalise user query before passing it to the memory retriever."""

    def run(self, query: str, **kwargs: object) -> HookResult:
        logger.debug(
            "pre_mem_fetch: normalising query",
            extra={"data": {"query_preview": query[:QUERY_LOG_PREVIEW_LENGTH]}},
        )

        cleaned = re.sub(r"[^a-zA-Z0-9\s']", "", query)
        cleaned = cleaned.lower()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        logger.debug(
            "pre_mem_fetch: done",
            extra={"data": {"result_preview": cleaned[:100]}},
        )
        return HookResult(success=True, message=cleaned)
