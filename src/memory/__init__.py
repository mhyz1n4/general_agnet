"""
The memory system package for managing agent long-term and short-term memory.

This package provides components for persisting (storage), organizing (indexing),
and searching (retrieval) information to provide context to the agent.

Sub-packages:
    file_system: JSON-file backed long-term storage, indexing, and retrieval.
    redis:       Redis-backed short-term session storage (optional dependency).
"""

from .base import BaseStorage, BaseIndexer, BaseRetriever, SearchResult
from .manager import MemoryManager
from .provider import (
    MemoryItem,
    MemoryProvider,
    MemoryType,
    Message,
    ReasoningContext,
    SearchFilters,
    Summary,
)
from .stub_provider import StubMemoryProvider

__all__ = [
    "BaseStorage",
    "BaseIndexer",
    "BaseRetriever",
    "SearchResult",
    "MemoryManager",
    "MemoryProvider",
    "MemoryItem",
    "MemoryType",
    "Message",
    "ReasoningContext",
    "SearchFilters",
    "Summary",
    "StubMemoryProvider",
]

# RedisStorage is an optional dependency — only exported when redis is installed.
try:
    from .redis import RedisStorage
    __all__.append("RedisStorage")
except ImportError:
    pass
