"""Unit tests for MemoryManager extensions: get_context_with_keys, dedup, DLQ integration."""

import pytest
from unittest.mock import MagicMock, call, patch

from src.memory.manager import MemoryManager
from src.memory.base import SearchResult


@pytest.fixture
def components():
    storage = MagicMock()
    indexer = MagicMock()
    indexer._load_index.return_value = {}
    retriever = MagicMock()
    return storage, indexer, retriever


@pytest.fixture
def mm(components):
    storage, indexer, retriever = components
    return MemoryManager(storage=storage, indexer=indexer, retriever=retriever)


# ---------------------------------------------------------------------------
# get_context_with_keys()
# ---------------------------------------------------------------------------

def test_get_context_with_keys_returns_tuple(mm, components):
    _, _, retriever = components
    retriever.search.return_value = [
        SearchResult(key="k1", content="fact one", relevance_score=1.0),
        SearchResult(key="k2", content="fact two", relevance_score=0.5),
    ]
    ctx, keys = mm.get_context_with_keys("test query")
    assert "k1" in ctx
    assert "k2" in ctx
    assert keys == ["k1", "k2"]


def test_get_context_with_keys_empty_query(mm):
    ctx, keys = mm.get_context_with_keys("")
    assert ctx == ""
    assert keys == []


def test_get_context_with_keys_no_results(mm, components):
    _, _, retriever = components
    retriever.search.return_value = []
    ctx, keys = mm.get_context_with_keys("something")
    assert ctx == ""
    assert keys == []


def test_get_context_backward_compat(mm, components):
    _, _, retriever = components
    retriever.search.return_value = [SearchResult(key="k1", content="c1", relevance_score=1.0)]
    result = mm.get_context("query")
    assert isinstance(result, str)
    assert "k1" in result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_dedup_same_content_skips_second_save(components):
    storage, indexer, retriever = components
    # First save: index empty
    indexer._load_index.return_value = {}
    mm = MemoryManager(storage=storage, indexer=indexer, retriever=retriever)
    mm.save_message("m1", "hello world", {"type": "semantic"})

    # Second save: index now contains the hash from first save
    import hashlib
    content_hash = hashlib.sha256("hello world".encode()).hexdigest()[:16]
    indexer._load_index.return_value = {
        "m1": {"metadata": {"content_hash": content_hash}}
    }
    mm.save_message("m2", "hello world", {"type": "semantic"})

    # storage.save should only have been called once
    assert storage.save.call_count == 1


def test_dedup_different_content_both_saved(components):
    storage, indexer, retriever = components
    indexer._load_index.return_value = {}
    mm = MemoryManager(storage=storage, indexer=indexer, retriever=retriever)
    mm.save_message("m1", "content A", {})
    mm.save_message("m2", "content B", {})
    assert storage.save.call_count == 2


# ---------------------------------------------------------------------------
# DLQ integration
# ---------------------------------------------------------------------------

def test_save_failure_enqueues_to_dlq(components):
    storage, indexer, retriever = components
    indexer._load_index.return_value = {}
    storage.save.side_effect = OSError("disk full")

    dlq = MagicMock()
    mm = MemoryManager(storage=storage, indexer=indexer, retriever=retriever, dlq=dlq)

    with pytest.raises(OSError):
        mm.save_message("m1", "content", {"type": "semantic"})

    dlq.enqueue.assert_called_once()
    args = dlq.enqueue.call_args[0]
    assert args[0] == "m1"


def test_save_failure_without_dlq_raises(components):
    storage, indexer, retriever = components
    indexer._load_index.return_value = {}
    storage.save.side_effect = OSError("disk full")

    mm = MemoryManager(storage=storage, indexer=indexer, retriever=retriever)
    with pytest.raises(OSError):
        mm.save_message("m1", "content", {})
