"""
Unit tests for MemoryProvider protocol and StubMemoryProvider.

StubMemoryProvider is the ground-truth reference implementation: its behaviour
defines what every other MemoryProvider MUST do. Tests here also verify the
runtime_checkable protocol so provider swaps are type-safe.
"""

from __future__ import annotations

import pytest

from src.memory.provider import (
    MemoryItem,
    MemoryProvider,
    SearchFilters,
    Summary,
)
from src.memory.stub_provider import StubMemoryProvider


@pytest.fixture
def provider() -> StubMemoryProvider:
    """Return a fresh in-memory ``StubMemoryProvider`` for each test."""
    return StubMemoryProvider()


class TestProtocolConformance:
    """StubMemoryProvider must satisfy the runtime protocol."""

    def test_isinstance_checks_pass(self, provider: StubMemoryProvider) -> None:
        """``StubMemoryProvider`` must pass a runtime ``isinstance`` check against ``MemoryProvider``."""
        assert isinstance(provider, MemoryProvider)


class TestSaveAndSearch:
    """save() must return a valid MemoryItem; search() must find it."""

    def test_save_returns_memory_item(self, provider: StubMemoryProvider) -> None:
        """``save`` returns a fully populated ``MemoryItem`` mirroring its inputs."""
        item = provider.save("user prefers Python", type="semantic", topic="prefs")
        assert isinstance(item, MemoryItem)
        assert item.content == "user prefers Python"
        assert item.type == "semantic"
        assert item.topic == "prefs"
        assert item.key  # non-empty

    def test_search_roundtrip(self, provider: StubMemoryProvider) -> None:
        """A saved item is discoverable via a substring query of its content."""
        provider.save("user prefers Python", type="semantic")
        results = provider.search("Python", SearchFilters(limit=5))
        assert len(results) == 1
        assert results[0].content == "user prefers Python"

    def test_search_empty_when_no_match(self, provider: StubMemoryProvider) -> None:
        """Queries that match no item return an empty list."""
        provider.save("user prefers Python", type="semantic")
        assert provider.search("JavaScript", SearchFilters(limit=5)) == []

    def test_search_respects_limit(self, provider: StubMemoryProvider) -> None:
        """``filters.limit`` caps the number of returned results."""
        for i in range(5):
            provider.save(f"fact {i} about cats", type="semantic")
        results = provider.search("cats", SearchFilters(limit=2))
        assert len(results) == 2

    def test_search_filters_by_type(self, provider: StubMemoryProvider) -> None:
        """``filters.types`` restricts results to the listed memory types."""
        provider.save("learned a fact", type="semantic")
        provider.save("ran a procedure", type="procedural")
        results = provider.search(
            "a", SearchFilters(limit=5, types=("semantic",))
        )
        assert len(results) == 1
        assert results[0].type == "semantic"

    def test_search_filters_by_topic(self, provider: StubMemoryProvider) -> None:
        """``filters.topics`` restricts results to the listed topics (empty query permitted)."""
        provider.save("foo", type="semantic", topic="work")
        provider.save("bar", type="semantic", topic="home")
        results = provider.search("", SearchFilters(limit=5, topics=("work",)))
        assert len(results) == 1
        assert results[0].topic == "work"


class TestCompact:
    """compact() produces a Summary containing all source messages."""

    def test_compact_returns_summary(self, provider: StubMemoryProvider) -> None:
        """The joined summary text contains every source message's content."""
        messages = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ]
        summary = provider.compact(messages)
        assert isinstance(summary, Summary)
        assert summary.source_count == 2
        assert "one" in summary.text and "two" in summary.text

    def test_compact_empty(self, provider: StubMemoryProvider) -> None:
        """Compacting an empty list returns an empty summary with ``source_count=0``."""
        summary = provider.compact([])
        assert summary.source_count == 0
        assert summary.text == ""


class TestCheckContext:
    """check_context() trims messages to fit token budget."""

    def test_under_budget_returns_all(self, provider: StubMemoryProvider) -> None:
        """When every message fits, the full list is returned unchanged."""
        msgs = [{"role": "user", "content": "hi"}]
        assert provider.check_context(msgs, budget_tokens=1000) == msgs

    def test_over_budget_trims_oldest(self, provider: StubMemoryProvider) -> None:
        """Oversized histories drop the oldest messages first."""
        msgs = [
            {"role": "user", "content": "x" * 100},
            {"role": "user", "content": "y" * 100},
            {"role": "user", "content": "z" * 100},
        ]
        # Budget ~50 tokens ≈ 200 chars → keeps last 2 messages.
        kept = provider.check_context(msgs, budget_tokens=50)
        assert len(kept) == 2
        assert kept[-1]["content"] == "z" * 100

    def test_zero_budget_drops_everything(self, provider: StubMemoryProvider) -> None:
        """A zero-token budget returns an empty list."""
        msgs = [{"role": "user", "content": "hi"}]
        assert provider.check_context(msgs, budget_tokens=0) == []
