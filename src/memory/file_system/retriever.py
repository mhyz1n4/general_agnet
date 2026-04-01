"""
Keyword-based retrieval implementation for the memory system.

This module provides the KeywordRetriever class, which searches the JSON index
to find and rank relevant content based on keyword matching.
"""

import json
import logging
import os
from typing import List, Dict, Any, Set
from ..base import BaseRetriever, SearchResult, BaseStorage

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

    def _load_index(self) -> Dict[str, Any]:
        """
        Load the index from the file system.

        Returns:
            A dictionary containing the index data, or an empty dictionary if the
            file does not exist or cannot be read.
        """
        if not os.path.exists(self.index_path):
            return {}
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return {}

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Perform a keyword search and return ranked results.

        Args:
            query: The user's search query string.
            limit: Maximum number of results to return. Defaults to 5.

        Returns:
            A list of SearchResult objects, sorted by their relevance score
            (number of matching keywords) in descending order.
        """
        if not query.strip():
            return []

        index: Dict[str, Any] = self._load_index()
        query_keywords: Set[str] = set(query.lower().split())
        results: List[SearchResult] = []

        for key, entry in index.items():
            entry_keywords: Set[str] = set(entry.get("keywords", []))
            # Calculate intersection for scoring
            matches: Set[str] = query_keywords.intersection(entry_keywords)
            score: int = len(matches)

            if score > 0:
                # Load content from storage
                # Assuming the key in the index matches the storage key
                stored_data: Any = self.storage.load(key)
                if stored_data is None:
                    continue

                # content is usually in the 'content' field for messages
                content: str = ""
                if isinstance(stored_data, dict):
                    content = stored_data.get("content", str(stored_data))
                else:
                    content = str(stored_data)

                results.append(SearchResult(
                    key=key,
                    content=content,
                    relevance_score=float(score),
                    metadata=entry.get("metadata", {})
                ))

        # Sort by relevance score descending
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]
