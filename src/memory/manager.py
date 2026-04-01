"""
Central coordination for the memory system.

This module provides the MemoryManager class, which orchestrates different storage,
indexing, and retrieval strategies to provide a unified interface for agent memory.
"""

import logging
from typing import List, Optional, Any, Dict
from .base import BaseStorage, BaseIndexer, BaseRetriever, SearchResult

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Orchestrates storage, indexing, and retrieval processes.

    MemoryManager provides high-level methods that unify access to various components
    of the memory system, simplifying content management for agents.

    Supports dual-backend operation: a long-term storage backend (e.g. file system)
    and an optional short-term session_storage backend (e.g. Redis). When a
    session_storage is provided, every saved message is mirrored there so the agent
    can leverage TTL-based expiry for active-session context.

    Query routing is driven by an optional classifier (RegexClassifier from src.query).
    An optional temporal_extractor (TemporalExtractor from src.query) provides date-range
    hints that will be passed to the retriever in v2.
    """

    def __init__(
        self,
        storage: BaseStorage,
        indexer: BaseIndexer,
        retriever: BaseRetriever,
        session_storage: Optional[BaseStorage] = None,
        classifier: Optional[Any] = None,
        temporal_extractor: Optional[Any] = None,
    ):
        """
        Initialize the MemoryManager with its core components.

        Args:
            storage: Long-term persistence backend (e.g. FileStorage).
            indexer: Manages the searchable index (e.g. JSONIndexer).
            retriever: Finds and ranks relevant content (e.g. KeywordRetriever).
            session_storage: Optional short-term backend (e.g. RedisStorage). Messages
                             are mirrored here alongside long-term storage. Operates
                             independently — no indexer or retriever required.
            classifier: Optional query intent classifier. Expected interface:
                        ``classifier.classify(query) -> obj`` where ``obj.value`` is one
                        of ``"recall_history"``, ``"current_session"``, ``"general_task"``.
                        Use ``RegexClassifier`` from ``src.query``.
            temporal_extractor: Optional temporal range extractor. Expected interface:
                                 ``extractor.extract(query) -> Optional[Tuple[datetime, datetime]]``.
                                 Use ``TemporalExtractor`` from ``src.query``.
                                 v1: used for debug logging only. v2: passed to retriever as filter.
        """
        self.storage: BaseStorage = storage
        self.indexer: BaseIndexer = indexer
        self.retriever: BaseRetriever = retriever
        self.session_storage: Optional[BaseStorage] = session_storage
        self.classifier = classifier
        self.temporal_extractor = temporal_extractor

    def save_message(
        self,
        message_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a message into long-term storage and update its search index.

        If a session_storage is configured the message is also mirrored there,
        giving the agent a fast TTL-scoped copy of recent context.

        Args:
            message_id: Unique identifier for the message.
            content: The text content of the message.
            metadata: Optional dictionary of additional metadata. Defaults to None.

        Raises:
            Exception: If any storage or indexing operation fails.
        """
        metadata = metadata or {}
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
            logger.debug(f"Message {message_id} saved and indexed.")
        except Exception as e:
            logger.error(f"Error saving message {message_id}: {str(e)}")
            raise

    def get_context(self, query: str, limit: int = 3) -> str:
        """
        Retrieve relevant context blocks from memory based on a query.

        When a classifier is provided the query intent drives routing:
        - ``current_session``: returns empty string — Redis has no index/retriever in v1.
          v2 will scan the Redis session window directly.
        - ``recall_history``: runs keyword search; temporal range is extracted and logged
          for v2 to use as a retriever filter.
        - ``general_task`` (default): runs keyword search directly.

        Formats results into a single string designed for inclusion in LLM prompts.

        Args:
            query: The search query string.
            limit: Maximum number of context blocks to return. Defaults to 3.

        Returns:
            A formatted string containing relevant context blocks, or an empty string
            if no matches are found.
        """
        if not query.strip():
            return ""

        if self.classifier is not None:
            intent = self.classifier.classify(query)
            intent_value: str = getattr(intent, "value", "")

            if intent_value == "current_session":
                # v1 limitation: Redis session storage has no indexer or retriever.
                # v2: scan Redis session window and rank by recency.
                logger.debug("CURRENT_SESSION intent — session search not supported in v1.")
                return ""

            if intent_value == "recall_history" and self.temporal_extractor is not None:
                time_range = self.temporal_extractor.extract(query)
                if time_range:
                    logger.debug(
                        f"Temporal range detected: {time_range[0]} to {time_range[1]}"
                    )
                    # v2: pass time_range as a filter to self.retriever.search()

        results: List[SearchResult] = self.retriever.search(query, limit=limit)
        if not results:
            return ""

        context_blocks: List[str] = [
            f"--- Context (Key: {r.key}) ---\n{r.content}\n"
            for r in results
        ]
        return "\n".join(context_blocks)
