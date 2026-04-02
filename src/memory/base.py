"""
Base abstract classes and data models for the memory system.

This module defines the core interfaces for storage, indexing, and retrieval,
ensuring consistency across different implementations of the memory system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """
    Data model representing a single search result from the memory system.

    Attributes:
        key: Unique identifier for the stored content.
        content: The actual text content associated with the key.
        relevance_score: Numerical value representing how relevant the result is to the query.
        metadata: Additional structured data associated with the content.
    """
    key: str
    content: str
    relevance_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseStorage(ABC):
    """
    Abstract base class for data storage implementations.
    """

    @abstractmethod
    def save(self, key: str, data: Any) -> None:
        """
        Save data to the storage backend.

        Args:
            key: Unique identifier for the data.
            data: The content to be stored. Can be any serializable object.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """
        Load data from the storage backend.

        Args:
            key: Unique identifier for the data.

        Returns:
            The stored data if found, otherwise None.
        """
        pass


class BaseIndexer(ABC):
    """
    Abstract base class for content indexing implementations.
    """

    @abstractmethod
    def add(self, key: str, content: str, metadata: Dict[str, Any]) -> None:
        """
        Index new content for future retrieval.

        Args:
            key: Unique identifier for the content.
            content: The text content to be indexed.
            metadata: Additional structured data associated with the content.

        Returns:
            None
        """
        pass

    @abstractmethod
    def update(self, key: str, content: str, metadata: Dict[str, Any]) -> None:
        """
        Update an existing index entry.

        Args:
            key: Unique identifier for the existing content.
            content: The updated text content.
            metadata: The updated metadata.

        Returns:
            None
        """
        pass

    @abstractmethod
    def find_by_content_hash(self, content_hash: str) -> Optional[str]:
        """
        Return the key of any existing entry whose metadata contains a matching
        content_hash, or None if no match is found.

        This is the correct abstraction for deduplication — MemoryManager
        calls this instead of inspecting the indexer's internal data structures.

        Args:
            content_hash: The hex content hash to search for.

        Returns:
            The key of the matching entry, or None.
        """
        pass

    @abstractmethod
    def touch(self, key: str) -> None:
        """
        Refresh the stored timestamp on an existing index entry without changing
        its content or keywords.  Called by MemoryManager when a duplicate write
        is detected so that the entry's recency stays current.

        Args:
            key: Unique identifier of the entry to update.

        Returns:
            None (silently no-ops if key does not exist).
        """
        pass


class BaseRetriever(ABC):
    """
    Abstract base class for memory retrieval implementations.
    """

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Search the index and return ranked results based on a query.

        Args:
            query: The search string to match against the index.
            limit: Maximum number of results to return. Defaults to 5.

        Returns:
            A list of SearchResult objects, ranked by relevance.
        """
        pass
