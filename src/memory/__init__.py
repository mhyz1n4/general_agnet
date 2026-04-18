"""
The memory system package for managing agent long-term memory.

V1.2: The ``MemoryProvider`` protocol is the primary interface (4 methods:
search, save, compact, check_context).  Session dialog is owned by Strands
Agent; the ``ReMeCompactionManager`` handles batch compaction and persistence.
"""

from .provider import (
    MemoryItem,
    MemoryProvider,
    MemoryType,
    Message,
    SearchFilters,
    Summary,
)
from .stub_provider import StubMemoryProvider

__all__ = [
    "MemoryProvider",
    "MemoryItem",
    "MemoryType",
    "Message",
    "SearchFilters",
    "Summary",
    "StubMemoryProvider",
]
