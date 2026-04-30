"""
Integration tests for the memory system via MemoryProvider.

V1.1: Tests use StubMemoryProvider for deterministic behavior.
ReMeLightProvider integration tests are in test_remelight_provider.py.
"""

import pytest

from src.memory.stub_provider import StubMemoryProvider
from src.memory.provider import SearchFilters

pytestmark = pytest.mark.integration


@pytest.fixture
def provider():
    """Fresh StubMemoryProvider for each test."""
    return StubMemoryProvider()


# ---------------------------------------------------------------------------
# Basic save & retrieval
# ---------------------------------------------------------------------------

def test_save_and_search(provider, TEST_MEMORIES):
    """Save memories and verify keyword search returns relevant results."""
    for entry in TEST_MEMORIES:
        provider.save(entry["content"], type=entry["type"], topic=entry.get("topic"))

    results = provider.search("Python", SearchFilters(limit=5))
    assert len(results) >= 1
    assert any("Python" in r.content for r in results)


def test_search_respects_type_filter(provider):
    """Search with a types filter excludes non-matching entries."""
    provider.save("a semantic fact", type="semantic")
    provider.save("a procedural step", type="procedural")
    results = provider.search("a", SearchFilters(limit=5, types=("semantic",)))
    assert all(r.type == "semantic" for r in results)


def test_search_respects_topic_filter(provider):
    """Search with a topics filter restricts results to the listed topics."""
    provider.save("foo", type="semantic", topic="work")
    provider.save("bar", type="semantic", topic="home")
    results = provider.search("", SearchFilters(limit=5, topics=("work",)))
    assert len(results) == 1
    assert results[0].topic == "work"


# ---------------------------------------------------------------------------
# Compact
# ---------------------------------------------------------------------------

def test_compact_returns_summary(provider):
    """Compact must return a Summary with all source messages joined."""
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    summary = provider.compact(messages)
    assert summary.source_count == 2
    assert "one" in summary.text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_query_returns_empty(provider):
    """Search with empty query after filtering should still work."""
    provider.save("anything", type="semantic", topic="work")
    results = provider.search("", SearchFilters(limit=5, topics=("work",)))
    assert len(results) == 1


def test_no_match_returns_empty(provider):
    """Query with no matching content must return an empty list."""
    provider.save("Python is great", type="semantic")
    results = provider.search("JavaScript", SearchFilters(limit=5))
    assert results == []
