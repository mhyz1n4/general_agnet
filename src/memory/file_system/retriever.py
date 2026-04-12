"""
Keyword-based retrieval implementation for the memory system.

This module provides the KeywordRetriever class, which searches the JSON index
to find and rank relevant content based on keyword matching.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Set
from ..base import BaseRetriever, SearchResult, BaseStorage
from ..types import IndexEntry, StorageRecord
from src.constants import DEFAULT_RETRIEVER_SEARCH_LIMIT, QUERY_LOG_PREVIEW_LENGTH

logger = logging.getLogger(__name__)


class KeywordRetriever(BaseRetriever):
    """
    Retriever that uses simple keyword matching and word frequency scoring.

    This class searches an index file for entries that match keywords from the query.
    Matching results are ranked based on the number of intersecting keywords
    and retrieved from storage to provide the full content.
    """

    def __init__(self, index_path: str, storage: BaseStorage):
        """
        Initialize the KeywordRetriever.

        Args:
            index_path: Absolute or relative path to the JSON index file.
            storage: Implementation of BaseStorage used to load the actual content.
        """
        self.index_path: str = index_path
        self.storage: BaseStorage = storage

    def _load_index(self) -> Dict[str, IndexEntry]:
        """
        Load the JSON index from disk without acquiring any locks.

        The retriever is read-only; it does not modify the index, so no
        locking is required.  The ``JSONIndexer`` is responsible for safe
        concurrent writes.

        Returns:
            The full index dict, or an empty dict if the file is absent or
            unparseable.
        """
        if not os.path.exists(self.index_path):
            return {}
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return {}

    def search(self, query: str, limit: int = DEFAULT_RETRIEVER_SEARCH_LIMIT) -> List[SearchResult]:
        """
        Perform a keyword search and return ranked results.

        First tries a compound search using all query tokens simultaneously.
        If that returns zero results, falls back to individual-token search
        (union of per-token hits) so that single-word matches are still found.

        Args:
            query: The user's search query string.
            limit: Maximum number of results to return. Defaults to 5.

        Returns:
            A list of SearchResult objects, sorted by their relevance score
            (number of matching keywords) in descending order.
        """
        if not query.strip():
            return []

        index: Dict[str, IndexEntry] = self._load_index()
        query_keywords: Set[str] = set(query.lower().split())

        results = self._score_and_load(query_keywords, index, limit)

        # Fallback: retry with individual tokens when compound query finds nothing.
        if not results and len(query_keywords) > 1:
            logger.debug(
                "retriever: compound query matched nothing, falling back to individual tokens",
                extra={"data": {"query": query[:QUERY_LOG_PREVIEW_LENGTH], "tokens": len(query_keywords)}},
            )
            seen_keys: Set[str] = set()
            fallback: List[SearchResult] = []
            for token in query_keywords:
                for result in self._score_and_load({token}, index, limit):
                    if result.key not in seen_keys:
                        seen_keys.add(result.key)
                        fallback.append(result)
            fallback.sort(key=lambda x: x.relevance_score, reverse=True)
            results = fallback[:limit]

        return results

    def _score_and_load(
        self,
        query_keywords: Set[str],
        index: Dict[str, IndexEntry],
        limit: int,
    ) -> List[SearchResult]:
        """
        Score every index entry by keyword intersection and load matching content.

        Entries with a score of zero (no intersection) are excluded.  For
        each matching entry the full content is loaded from storage; entries
        whose storage key resolves to ``None`` are silently skipped.

        Args:
            query_keywords: Set of lowercased query tokens to match against.
            index:          The full in-memory index dict.
            limit:          Maximum number of results to return.

        Returns:
            List of ``SearchResult`` objects sorted by ``relevance_score``
            descending, capped at *limit*.
        """
        results: List[SearchResult] = []

        for key, entry in index.items():
            entry_keywords: Set[str] = set(entry.get("keywords", []))
            matches: Set[str] = query_keywords.intersection(entry_keywords)
            score: int = len(matches)

            if score > 0:
                stored_data: Optional[StorageRecord] = self.storage.load(key)
                if stored_data is None:
                    continue

                content: str = ""
                if isinstance(stored_data, dict):
                    content = stored_data.get("content", str(stored_data))
                else:
                    content = str(stored_data)

                results.append(SearchResult(
                    key=key,
                    content=content,
                    relevance_score=float(score),
                    metadata=entry.get("metadata", {}),
                ))

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]
