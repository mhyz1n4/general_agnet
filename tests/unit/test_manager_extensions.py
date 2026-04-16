"""
Unit tests for MemoryManager extensions: get_context_with_keys, dedup, DLQ integration,
and session eviction.

All external dependencies (storage, indexer, retriever) are mocked so these tests run
without any filesystem I/O.
"""

from typing import Dict, Tuple
from unittest.mock import MagicMock, call, patch

import pytest

from src.memory.manager import MemoryManager
from src.memory.base import SearchResult


@pytest.fixture
def components() -> Tuple[MagicMock, MagicMock, MagicMock]:
    """
    Return a (storage, indexer, retriever) triple of MagicMocks.

    indexer.find_by_content_hash defaults to None (no duplicate found),
    so the first save_message call always proceeds to storage.
    """
    storage: MagicMock = MagicMock()
    indexer: MagicMock = MagicMock()
    indexer.find_by_content_hash.return_value = None
    retriever: MagicMock = MagicMock()
    return storage, indexer, retriever


@pytest.fixture
def mm(components: Tuple[MagicMock, MagicMock, MagicMock]) -> MemoryManager:
    """MemoryManager built from the default mocked components fixture."""
    storage, indexer, retriever = components
    return MemoryManager(storage=storage, indexer=indexer, retriever=retriever)


# ---------------------------------------------------------------------------
# get_context_with_keys()
# ---------------------------------------------------------------------------


def test_get_context_with_keys_returns_tuple(
    mm: MemoryManager,
    components: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Context string must include all retrieved entries; keys list must match their order."""
    _, _, retriever = components
    retriever.search.return_value = [
        SearchResult(key="k1", content="fact one", relevance_score=1.0),
        SearchResult(key="k2", content="fact two", relevance_score=0.5),
    ]
    ctx: str
    keys: list
    ctx, keys = mm.get_context_with_keys("test query")
    assert "fact one" in ctx
    assert "fact two" in ctx
    assert keys == ["k1", "k2"]


def test_get_context_with_keys_empty_query(mm: MemoryManager) -> None:
    """An empty query must return an empty context string and an empty keys list immediately."""
    ctx: str
    keys: list
    ctx, keys = mm.get_context_with_keys("")
    assert ctx == ""
    assert keys == []


def test_get_context_with_keys_no_results(
    mm: MemoryManager,
    components: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """When the retriever returns no results the context must be an empty string."""
    _, _, retriever = components
    retriever.search.return_value = []
    ctx: str
    keys: list
    ctx, keys = mm.get_context_with_keys("something")
    assert ctx == ""
    assert keys == []


def test_get_context_backward_compat(
    mm: MemoryManager,
    components: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """get_context() (deprecated wrapper) must return a plain string containing the content."""
    _, _, retriever = components
    retriever.search.return_value = [SearchResult(key="k1", content="c1", relevance_score=1.0)]
    result = mm.get_context("query")
    assert isinstance(result, str)
    assert "c1" in result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_dedup_same_content_skips_second_save(
    components: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Saving identical content twice must only call storage.save once; touch refreshes the ts."""
    storage, indexer, retriever = components
    mm: MemoryManager = MemoryManager(storage=storage, indexer=indexer, retriever=retriever)

    mm.save_message("m1", "hello world", {"type": "semantic"})
    assert storage.save.call_count == 1

    # Second save with identical content: hash resolves to the existing key
    indexer.find_by_content_hash.return_value = "m1"
    mm.save_message("m2", "hello world", {"type": "semantic"})

    assert storage.save.call_count == 1
    indexer.touch.assert_called_once_with("m1")


def test_dedup_different_content_both_saved(
    components: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Two messages with distinct content must each result in a separate storage.save call."""
    storage, indexer, retriever = components
    indexer.find_by_content_hash.return_value = None
    mm: MemoryManager = MemoryManager(storage=storage, indexer=indexer, retriever=retriever)
    mm.save_message("m1", "content A", {})
    mm.save_message("m2", "content B", {})
    assert storage.save.call_count == 2


# ---------------------------------------------------------------------------
# DLQ integration
# ---------------------------------------------------------------------------


def test_save_failure_enqueues_to_dlq(
    components: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """A storage.save() failure must enqueue the message to the DLQ and re-raise the error."""
    storage, indexer, retriever = components
    indexer.find_by_content_hash.return_value = None
    storage.save.side_effect = OSError("disk full")

    dlq: MagicMock = MagicMock()
    mm: MemoryManager = MemoryManager(storage=storage, indexer=indexer, retriever=retriever, dlq=dlq)

    with pytest.raises(OSError):
        mm.save_message("m1", "content", {"type": "semantic"})

    dlq.enqueue.assert_called_once()
    args = dlq.enqueue.call_args[0]
    assert args[0] == "m1"


def test_save_failure_without_dlq_raises(
    components: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Without a DLQ configured, a storage.save() failure must propagate as an OSError."""
    storage, indexer, retriever = components
    indexer.find_by_content_hash.return_value = None
    storage.save.side_effect = OSError("disk full")

    mm: MemoryManager = MemoryManager(storage=storage, indexer=indexer, retriever=retriever)
    with pytest.raises(OSError):
        mm.save_message("m1", "content", {})


# ---------------------------------------------------------------------------
# Session eviction
# ---------------------------------------------------------------------------


def _make_session_record(key: str, ts: str = "2026-04-03T00:00:00+00:00") -> dict:
    """Build a minimal episodic StorageRecord dict for use in eviction tests."""
    return {"id": key, "content": f"content of {key}", "metadata": {"type": "episodic", "timestamp": ts}}


def test_eviction_triggered_at_limit(
    components: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """When session_storage is at session_max_messages, the oldest entry must be evicted."""
    storage, indexer, retriever = components
    indexer.find_by_content_hash.return_value = None

    session_storage: MagicMock = MagicMock()
    session_storage.list_keys.return_value = ["old1", "old2", "old3"]
    session_storage.load.side_effect = lambda k: _make_session_record(
        k, ts=f"2026-04-03T0{['old1','old2','old3'].index(k)}:00:00+00:00"
    )

    mm: MemoryManager = MemoryManager(
        storage=storage, indexer=indexer, retriever=retriever,
        session_storage=session_storage, session_max_messages=3,
    )
    mm.save_message("new_msg", "new content", {"type": "episodic"})

    session_storage.delete.assert_called_once_with("old1")
    assert mm.eviction_count == 1


def test_no_eviction_below_limit(
    components: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """When session_storage is below the message limit, no eviction must occur."""
    storage, indexer, retriever = components
    indexer.find_by_content_hash.return_value = None

    session_storage: MagicMock = MagicMock()
    session_storage.list_keys.return_value = ["m1", "m2"]

    mm: MemoryManager = MemoryManager(
        storage=storage, indexer=indexer, retriever=retriever,
        session_storage=session_storage, session_max_messages=3,
    )
    mm.save_message("m3", "content", {})

    session_storage.delete.assert_not_called()
    assert mm.eviction_count == 0


def test_evicted_entry_written_to_long_term(
    components: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """The evicted session entry must be written to long-term storage before deletion."""
    storage, indexer, retriever = components
    indexer.find_by_content_hash.return_value = None

    session_storage: MagicMock = MagicMock()
    session_storage.list_keys.return_value = ["oldest"]
    evicted: dict = _make_session_record("oldest", ts="2025-01-01T00:00:00+00:00")
    session_storage.load.return_value = evicted

    mm: MemoryManager = MemoryManager(
        storage=storage, indexer=indexer, retriever=retriever,
        session_storage=session_storage, session_max_messages=1,
    )
    mm.save_message("new", "new content", {})

    eviction_saves = [c for c in storage.save.call_args_list if c[0][0] == "oldest"]
    assert len(eviction_saves) == 1


def test_eviction_count_increments(
    components: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """eviction_count must increment once per eviction across multiple save_message calls."""
    storage, indexer, retriever = components
    indexer.find_by_content_hash.return_value = None

    session_storage: MagicMock = MagicMock()
    session_storage.list_keys.return_value = ["x"]
    session_storage.load.return_value = _make_session_record("x")

    mm: MemoryManager = MemoryManager(
        storage=storage, indexer=indexer, retriever=retriever,
        session_storage=session_storage, session_max_messages=1,
    )
    mm.save_message("a", "content a", {})
    indexer.find_by_content_hash.return_value = None
    session_storage.list_keys.return_value = ["x"]
    mm.save_message("b", "content b", {})
    assert mm.eviction_count == 2
