"""
Central coordination for the memory system.

This module provides the MemoryManager class, which orchestrates different storage,
indexing, and retrieval strategies to provide a unified interface for agent memory.
"""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple

from .base import BaseStorage, BaseIndexer, BaseRetriever, SearchResult
from .types import MemoryMetadata, StorageRecord
from src.constants import (
    CONTENT_HASH_LENGTH,
    CONTEXT_BLOCK_DELIMITER,
    DEFAULT_RETRIEVAL_LIMIT,
    DEFAULT_SESSION_MAX_MESSAGES,
    INTENT_CURRENT_SESSION,
    INTENT_RECALL_HISTORY,
)

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
        classifier: Optional["RegexClassifier"] = None,
        temporal_extractor: Optional["TemporalExtractor"] = None,
        dlq: Optional["DeadLetterQueue"] = None,
        session_max_messages: int = DEFAULT_SESSION_MAX_MESSAGES,
        session_id: Optional[str] = None,
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
            session_max_messages: Maximum messages to keep in session_storage before
                evicting the oldest entry to long-term storage.
            session_id: Current session identifier. When provided, session eviction
                only considers keys whose stored metadata.session_id matches, so
                concurrent sessions do not evict each other's entries.
        """
        self.storage: BaseStorage = storage
        self.indexer: BaseIndexer = indexer
        self.retriever: BaseRetriever = retriever
        self.session_storage: Optional[BaseStorage] = session_storage
        self.classifier = classifier
        self.temporal_extractor = temporal_extractor
        self.dlq = dlq
        self.session_max_messages = session_max_messages
        self.session_id: Optional[str] = session_id
        self.eviction_count: int = 0

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_message(
        self,
        message_id: str,
        content: str,
        metadata: Optional[MemoryMetadata] = None,
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
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:CONTENT_HASH_LENGTH]
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

        data: StorageRecord = {
            "id": message_id,
            "content": content,
            "metadata": metadata,
        }
        try:
            self.storage.save(message_id, data)
            self.indexer.add(message_id, content, metadata)
            if self.session_storage is not None:
                self._enforce_session_limit()
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
    # Session eviction
    # ------------------------------------------------------------------

    def _enforce_session_limit(self) -> None:
        """
        Evict the oldest session message to long-term storage when the session
        window is at capacity. Runs synchronously before each new session write.

        When ``session_id`` is set, only keys whose stored metadata.session_id
        matches are counted and evicted — this prevents concurrent sessions from
        evicting each other's entries from shared session storage.
        """
        if self.session_storage is None:
            return
        try:
            all_keys: List[str] = self.session_storage.list_keys()

            # Collect only the keys that belong to the current session (when known).
            # Keys with no matching session_id are left untouched so concurrent
            # sessions do not interfere with each other.
            session_keys: List[str] = []
            for key in all_keys:
                if self.session_id is not None:
                    record = self.session_storage.load(key)
                    if record is None:
                        continue
                    if record.get("metadata", {}).get("session_id") != self.session_id:
                        continue
                session_keys.append(key)

            if len(session_keys) < self.session_max_messages:
                return

            # Find the oldest key by timestamp in stored metadata.
            # Keys with no timestamp sort to the front (empty string < any ISO-8601).
            oldest_key: Optional[str] = None
            oldest_ts: str = "~"  # "~" sorts after all valid ISO-8601 strings
            for key in session_keys:
                record = self.session_storage.load(key)
                if record is None:
                    continue
                ts = record.get("metadata", {}).get("timestamp", "")
                if ts < oldest_ts:
                    oldest_ts = ts
                    oldest_key = key

            if oldest_key is None:
                return

            record = self.session_storage.load(oldest_key)
            if record is not None:
                # Write to long-term storage directly — bypass dedup and DLQ
                # to avoid recursion; this is a best-effort eviction path.
                try:
                    self.storage.save(oldest_key, record)
                    self.indexer.add(oldest_key, record["content"], record["metadata"])
                except Exception as exc:
                    logger.error(
                        "session_eviction: long-term write failed",
                        extra={"data": {"key": oldest_key, "error": str(exc)}},
                    )
            self.session_storage.delete(oldest_key)
            self.eviction_count += 1
            logger.debug(
                "session_eviction: evicted oldest entry to long-term storage",
                extra={"data": {"key": oldest_key, "eviction_count": self.eviction_count}},
            )
        except Exception as exc:
            logger.error(
                "session_eviction: unexpected error",
                extra={"data": {"error": str(exc)}},
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_context_with_keys(self, query: str, limit: int = DEFAULT_RETRIEVAL_LIMIT) -> Tuple[str, List[str]]:
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

            if intent_value == INTENT_CURRENT_SESSION:
                logger.debug("CURRENT_SESSION intent — session search not supported in v1.")
                return "", []

            if intent_value == INTENT_RECALL_HISTORY and self.temporal_extractor is not None:
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
            f"{CONTEXT_BLOCK_DELIMITER}\n{r.content}\n"
            for r in results
        ]
        keys = [r.key for r in results]
        return "\n".join(context_blocks), keys

    def get_context(self, query: str, limit: int = DEFAULT_RETRIEVAL_LIMIT) -> str:
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
