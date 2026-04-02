"""
Central coordination for the memory system.

This module provides the MemoryManager class, which orchestrates different storage,
indexing, and retrieval strategies to provide a unified interface for agent memory.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseStorage, BaseIndexer, BaseRetriever, SearchResult

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Orchestrates storage, indexing, and retrieval processes.

    Supports dual-backend operation: a long-term storage backend (e.g. file system)
    and an optional short-term session_storage backend (e.g. Redis). When a
    session_storage is provided, every saved message is mirrored there so the agent
    can leverage TTL-based expiry for active-session context.

    Query routing is driven by an optional classifier (RegexClassifier from src.query).
    An optional temporal_extractor (TemporalExtractor from src.query) provides date-range
    hints that will be passed to the retriever in v2.

    Content-hash deduplication: before each write, a 16-char SHA-256 prefix is checked
    against the index. Duplicate content is skipped (timestamp updated instead).

    DLQ integration: when a save fails the entry is forwarded to an optional
    DeadLetterQueue for background retry.
    """

    def __init__(
        self,
        storage: BaseStorage,
        indexer: BaseIndexer,
        retriever: BaseRetriever,
        session_storage: Optional[BaseStorage] = None,
        classifier: Optional[Any] = None,
        temporal_extractor: Optional[Any] = None,
        dlq: Optional[Any] = None,
    ):
        """
        Initialize the MemoryManager with its core components.

        Args:
            storage: Long-term persistence backend (e.g. FileStorage, TypedMarkdownStorage).
            indexer: Manages the searchable index (e.g. JSONIndexer).
            retriever: Finds and ranks relevant content (e.g. KeywordRetriever).
            session_storage: Optional short-term backend (e.g. RedisStorage).
            classifier: Optional query intent classifier (RegexClassifier).
            temporal_extractor: Optional temporal range extractor (TemporalExtractor).
            dlq: Optional DeadLetterQueue for failed save retries.
        """
        self.storage: BaseStorage = storage
        self.indexer: BaseIndexer = indexer
        self.retriever: BaseRetriever = retriever
        self.session_storage: Optional[BaseStorage] = session_storage
        self.classifier = classifier
        self.temporal_extractor = temporal_extractor
        self.dlq = dlq

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_message(
        self,
        message_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a message into long-term storage and update its search index.

        Performs content-hash deduplication before writing. If a session_storage is
        configured, the message is mirrored there after the primary write.

        On failure, forwards the entry to the DLQ (if configured) then re-raises.

        Args:
            message_id: Unique identifier for the message.
            content: The text content of the message.
            metadata: Optional dictionary of additional metadata.

        Raises:
            Exception: If any storage or indexing operation fails.
        """
        metadata = metadata or {}

        # Content-hash deduplication — uses the BaseIndexer interface only,
        # no access to implementation-private methods.
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        try:
            existing_key = self.indexer.find_by_content_hash(content_hash)
            if existing_key is not None:
                # Refresh the timestamp so the entry stays current in any
                # recency-based ranking, as documented in the class docstring.
                self.indexer.touch(existing_key)
                logger.debug(
                    "save_message dedup: duplicate content, timestamp refreshed",
                    extra={"data": {"message_id": message_id, "matches": existing_key}},
                )
                return
        except Exception:
            # Dedup is best-effort; proceed with the write on any indexer error.
            pass

        metadata = {**metadata, "content_hash": content_hash}

        data: Dict[str, Any] = {
            "id": message_id,
            "content": content,
            "metadata": metadata,
        }
        try:
            self.storage.save(message_id, data)
            self.indexer.add(message_id, content, metadata)
            if self.session_storage is not None:
                self.session_storage.save(message_id, data)
            logger.debug(
                "save_message: saved and indexed",
                extra={"data": {"message_id": message_id}},
            )
        except Exception as exc:
            logger.error(
                "save_message: failed",
                extra={"data": {"message_id": message_id, "error": str(exc)}},
            )
            if self.dlq is not None:
                self.dlq.enqueue(message_id, content, metadata, str(exc))
            raise

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_context_with_keys(self, query: str, limit: int = 3) -> Tuple[str, List[str]]:
        """
        Retrieve relevant context blocks from memory based on a query.

        Returns:
            Tuple of (formatted_context_string, list_of_retrieved_keys).
        """
        if not query.strip():
            return "", []

        if self.classifier is not None:
            intent = self.classifier.classify(query)
            intent_value: str = getattr(intent, "value", "")

            if intent_value == "current_session":
                logger.debug("CURRENT_SESSION intent — session search not supported in v1.")
                return "", []

            if intent_value == "recall_history" and self.temporal_extractor is not None:
                time_range = self.temporal_extractor.extract(query)
                if time_range:
                    logger.debug(
                        "Temporal range detected",
                        extra={"data": {"start": str(time_range[0]), "end": str(time_range[1])}},
                    )
                    # v2: pass time_range as a filter to self.retriever.search()

        results: List[SearchResult] = self.retriever.search(query, limit=limit)
        if not results:
            return "", []

        context_blocks: List[str] = [
            f"--- Context (Key: {r.key}) ---\n{r.content}\n"
            for r in results
        ]
        keys = [r.key for r in results]
        return "\n".join(context_blocks), keys

    def get_context(self, query: str, limit: int = 3) -> str:
        """
        Retrieve relevant context blocks from memory based on a query.

        Backward-compatible wrapper around get_context_with_keys().

        Args:
            query: The search query string.
            limit: Maximum number of context blocks to return.

        Returns:
            A formatted string containing relevant context blocks, or "" if none found.
        """
        context, _ = self.get_context_with_keys(query, limit)
        return context
