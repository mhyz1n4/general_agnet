"""
Integration tests for ReMeLightProvider.

These tests boot a real ReMeLight instance (SQLite FTS, no vectors, stub LLM)
on a ``tmp_path`` working directory. They verify end-to-end behavior through
the ``MemoryProvider`` interface: save/search round-trip, session persistence,
compact, check_context.

The ReMeLight file watcher indexes saved .md files asynchronously. Tests use
a short poll loop to wait for indexing before asserting on search results.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pytest

from src.memory.provider import MemoryProvider, SearchFilters, Summary
from src.memory.remelight.provider import ReMeLightProvider


WATCHER_INDEX_TIMEOUT_SECONDS = 10.0
WATCHER_POLL_INTERVAL_SECONDS = 0.5


@pytest.fixture
def provider(tmp_path: Path) -> Iterator[ReMeLightProvider]:
    """
    Spin up a real ``ReMeLightProvider`` rooted at ``tmp_path/reme`` and
    tear it down at test end.

    Yields:
        A started ``ReMeLightProvider`` bound to session ``"test-session"``.
    """
    p = ReMeLightProvider(memory_root=str(tmp_path / "reme"), session_id="test-session")
    try:
        yield p
    finally:
        p.close()


def _search_until_found(
    provider: ReMeLightProvider,
    query: str,
    filters: SearchFilters,
    timeout: float = WATCHER_INDEX_TIMEOUT_SECONDS,
) -> list:
    """
    Poll ``provider.search`` until it returns at least one result or the
    watcher-index timeout elapses.

    Args:
        provider: Running ``ReMeLightProvider`` to query.
        query:    Search string passed through to ``search``.
        filters:  Structured filters passed through to ``search``.
        timeout:  Maximum seconds to poll before giving up.

    Returns:
        The first non-empty result list, or ``[]`` if the timeout expires.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        results = provider.search(query, filters)
        if results:
            return results
        time.sleep(WATCHER_POLL_INTERVAL_SECONDS)
    return []


class TestProtocolConformance:
    """ReMeLightProvider must satisfy the runtime protocol."""

    def test_isinstance_memory_provider(self, provider: ReMeLightProvider) -> None:
        """``ReMeLightProvider`` must pass a runtime ``isinstance`` check against ``MemoryProvider``."""
        assert isinstance(provider, MemoryProvider)


class TestSaveAndSearch:
    """Saved memories are discoverable by FTS after the watcher indexes them."""

    def test_save_writes_file(self, provider: ReMeLightProvider, tmp_path: Path) -> None:
        """``save`` materialises a frontmattered ``.md`` file under ``memory/`` with the expected fields."""
        provider.save("user prefers Python", type="semantic", topic="prefs")
        memory_dir = tmp_path / "reme" / "memory"
        assert memory_dir.exists()
        files = list(memory_dir.glob("*.md"))
        assert len(files) == 1
        text = files[0].read_text(encoding="utf-8")
        assert "type: semantic" in text
        assert "topic: prefs" in text
        assert "user prefers Python" in text

    def test_search_roundtrip(self, provider: ReMeLightProvider) -> None:
        """After the watcher indexes a save, ``search`` can find the entry by keyword."""
        provider.save("user prefers Python for scripting", type="semantic", topic="prefs")
        results = _search_until_found(provider, "Python", SearchFilters(limit=5))
        assert len(results) >= 1
        assert results[0].type == "semantic"

    def test_search_empty_query_returns_nothing(
        self, provider: ReMeLightProvider
    ) -> None:
        """An empty query short-circuits to ``[]`` even when items exist."""
        provider.save("anything", type="semantic")
        assert provider.search("", SearchFilters(limit=5)) == []

    def test_search_filters_by_type(self, provider: ReMeLightProvider) -> None:
        """A ``types`` filter excludes entries whose type is not in the whitelist."""
        provider.save("a semantic fact", type="semantic")
        provider.save("a procedural step", type="procedural")
        # Wait for index, then query with a type filter that excludes one.
        results = _search_until_found(
            provider, "a", SearchFilters(limit=5, types=("semantic",))
        )
        assert all(r.type == "semantic" for r in results)


class TestSessionHistory:
    """Session turns persist to dialog/ and round-trip by session_id."""

    def test_single_session_roundtrip(self, provider: ReMeLightProvider) -> None:
        """Turns saved under one session return in insertion order from ``get_session_history``."""
        provider.save_session_turn(
            {"role": "user", "content": "hello", "session_id": "s-alpha"}
        )
        provider.save_session_turn(
            {"role": "assistant", "content": "hi", "session_id": "s-alpha"}
        )
        history = provider.get_session_history("s-alpha")
        assert len(history) == 2
        assert history[0]["content"] == "hello"
        assert history[1]["content"] == "hi"

    def test_sessions_isolated(self, provider: ReMeLightProvider) -> None:
        """Turns saved under different sessions are returned independently; unknown sessions yield ``[]``."""
        provider.save_session_turn(
            {"role": "user", "content": "in a", "session_id": "s-a"}
        )
        provider.save_session_turn(
            {"role": "user", "content": "in b", "session_id": "s-b"}
        )
        assert len(provider.get_session_history("s-a")) == 1
        assert len(provider.get_session_history("s-b")) == 1
        assert provider.get_session_history("s-missing") == []


class TestCompact:
    """compact() delegates to ReMeLight; with stub LLM, text is empty."""

    def test_compact_preserves_source_count(self, provider: ReMeLightProvider) -> None:
        """``compact`` returns ``source_count=len(msgs)`` even when the stub LLM yields empty text."""
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        summary = provider.compact(msgs)
        assert isinstance(summary, Summary)
        assert summary.source_count == 2
        # Stub LLM returns empty string — this is documented V1.1 behaviour.
        assert summary.text == ""

    def test_compact_empty_messages(self, provider: ReMeLightProvider) -> None:
        """Compacting an empty list yields an empty summary with ``source_count=0``."""
        summary = provider.compact([])
        assert summary.source_count == 0
        assert summary.text == ""


class TestCheckContext:
    """check_context() delegates to ReMeLight's ContextChecker."""

    def test_small_budget_drops_messages(self, provider: ReMeLightProvider) -> None:
        """A budget far smaller than content size drops every message."""
        msgs = [{"role": "user", "content": "x" * 1000}]
        kept = provider.check_context(msgs, budget_tokens=10)
        # Budget far below content size — ContextChecker returns empty keep set.
        assert kept == []

    def test_large_budget_keeps_everything(self, provider: ReMeLightProvider) -> None:
        """A generous budget keeps every input message intact."""
        msgs = [
            {"role": "user", "content": "short one"},
            {"role": "assistant", "content": "short two"},
        ]
        kept = provider.check_context(msgs, budget_tokens=100_000)
        assert len(kept) == 2


class TestPreReasoningHook:
    """Hook returns context unchanged when under budget."""

    def test_passthrough_with_generous_budget(
        self, provider: ReMeLightProvider
    ) -> None:
        """With a generous budget the hook leaves ``context.messages`` untouched."""
        from src.memory.provider import ReasoningContext

        ctx = ReasoningContext(
            messages=[{"role": "user", "content": "hi"}],
            memory_items=[],
            budget_tokens=100_000,
        )
        out = provider.pre_reasoning_hook(ctx)
        assert len(out.messages) == 1
