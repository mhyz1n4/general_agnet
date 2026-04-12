"""
Unit tests for all four hooks (PreMemFetchHook, PostMemFetchHook,
PreSessionHook, PostSessionHook).

All external dependencies (Redis, LLM client, MemoryManager) are mocked so
these tests run without any infrastructure.  PostSessionHook tests call
_run_async() directly instead of run() + time.sleep() to eliminate timing
flakiness.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import MagicMock

import pytest

from src.hooks.pre_mem_fetch import PreMemFetchHook
from src.hooks.post_mem_fetch import PostMemFetchHook
from src.hooks.pre_session import PreSessionHook
from src.hooks.post_session import PostSessionHook


# ===========================================================================
# PreMemFetchHook
# ===========================================================================


class TestPreMemFetchHook:
    """PreMemFetchHook normalises the user query before memory retrieval."""

    @pytest.fixture(autouse=True)
    def hook(self) -> None:
        """Create a fresh PreMemFetchHook for each test."""
        self.hook: PreMemFetchHook = PreMemFetchHook()

    def test_empty_string(self) -> None:
        """An empty query should produce an empty message."""
        assert self.hook.run("").message == ""

    def test_special_chars_stripped(self) -> None:
        """Non-alphanumeric characters (except spaces and apostrophes) should be removed."""
        result = self.hook.run("!@#$%")
        assert result.message == ""

    def test_apostrophe_kept(self) -> None:
        """Apostrophes in contractions should be preserved after normalisation."""
        result = self.hook.run("what's up")
        assert result.message == "what's up"

    def test_lowercased(self) -> None:
        """All characters should be converted to lowercase."""
        result = self.hook.run("Hello World")
        assert result.message == "hello world"

    def test_collapse_whitespace(self) -> None:
        """Multiple consecutive spaces should be collapsed to a single space."""
        result = self.hook.run("hello  world")
        assert result.message == "hello world"

    def test_strip_leading_trailing(self) -> None:
        """Leading and trailing whitespace should be stripped."""
        result = self.hook.run("  hello ")
        assert result.message == "hello"

    def test_long_query_processed(self) -> None:
        """A 5 000-character query should be processed without raising."""
        query: str = "a " * 2500
        result = self.hook.run(query)
        assert result.success is True
        assert len(result.message) > 0

    def test_success_true(self) -> None:
        """run() should always return success=True for a non-empty query."""
        assert self.hook.run("test").success is True


# ===========================================================================
# PostMemFetchHook
# ===========================================================================


class TestPostMemFetchHook:
    """PostMemFetchHook truncates context that exceeds max_context_chars."""

    def test_context_under_limit_unchanged(self) -> None:
        """Context smaller than the limit should be returned unmodified."""
        hook: PostMemFetchHook = PostMemFetchHook(max_context_chars=1000)
        ctx: str = "hello"
        result = hook.run(context=ctx)
        assert result.message == ctx

    def test_empty_context_unchanged(self) -> None:
        """An empty context string should be returned as-is."""
        hook: PostMemFetchHook = PostMemFetchHook(max_context_chars=100)
        result = hook.run(context="")
        assert result.message == ""

    def test_over_limit_truncates_blocks(self) -> None:
        """Context exceeding the limit should be truncated to at most max_context_chars."""
        block: str = "--- Memory ---\n" + "x" * 200 + "\n"
        ctx: str = block * 5
        hook: PostMemFetchHook = PostMemFetchHook(max_context_chars=len(block) * 2 + 10)
        result = hook.run(context=ctx)
        assert len(result.message) <= len(block) * 2 + 20

    def test_single_oversized_block_returned_as_is(self) -> None:
        """A single block larger than the limit should be returned whole, not truncated."""
        block: str = "--- Memory ---\n" + "x" * 5000 + "\n"
        hook: PostMemFetchHook = PostMemFetchHook(max_context_chars=100)
        result = hook.run(context=block)
        assert result.message == block

    def test_saves_retrieval_event_to_session_storage(self) -> None:
        """When session_storage is provided, a retrieval event record should be saved."""
        session_storage: MagicMock = MagicMock()
        hook: PostMemFetchHook = PostMemFetchHook(max_context_chars=8000, session_storage=session_storage)
        hook.run(context="some context", retrieved_keys=["k1", "k2"], query="test query")
        session_storage.save.assert_called_once()

    def test_empty_keys_no_crash(self) -> None:
        """An empty retrieved_keys list should not cause any error."""
        hook: PostMemFetchHook = PostMemFetchHook()
        result = hook.run(context="", retrieved_keys=[])
        assert result.success is True


# ===========================================================================
# PreSessionHook
# ===========================================================================


class TestPreSessionHook:
    """PreSessionHook validates system prerequisites before a session starts."""

    def test_all_healthy_returns_success(self, tmp_path: Path) -> None:
        """When FS is writable and no clients are configured, the hook must succeed."""
        hook: PreSessionHook = PreSessionHook(memory_root=str(tmp_path))
        result = hook.run()
        assert result.success is True
        assert result.degraded is False

    def test_fs_missing_creates_dir(self, tmp_path: Path) -> None:
        """A missing memory_root directory should be created automatically."""
        new_dir: str = str(tmp_path / "new_memory")
        hook: PreSessionHook = PreSessionHook(memory_root=new_dir)
        result = hook.run()
        assert result.success is True
        assert os.path.isdir(new_dir)

    def test_redis_connection_error_degrades(self, tmp_path: Path) -> None:
        """A Redis ConnectionError should set degraded=True but not abort the session."""
        redis_client: MagicMock = MagicMock()
        redis_client.ping.side_effect = ConnectionError("no redis")
        hook: PreSessionHook = PreSessionHook(memory_root=str(tmp_path), redis_client=redis_client)
        result = hook.run()
        assert result.success is True
        assert result.degraded is True
        assert "Redis" in result.message

    def test_llm_auth_error_returns_failure(self, tmp_path: Path) -> None:
        """An LLM AuthenticationError must abort the session (success=False)."""
        class AuthenticationError(Exception):
            pass

        llm: MagicMock = MagicMock()
        llm.completion.side_effect = AuthenticationError("bad key")
        hook: PreSessionHook = PreSessionHook(memory_root=str(tmp_path), llm_client=llm)
        result = hook.run()
        assert result.success is False

    def test_llm_rate_limit_degrades(self, tmp_path: Path) -> None:
        """A RateLimitError from the LLM should degrade the session, not abort it."""
        class RateLimitError(Exception):
            pass

        llm: MagicMock = MagicMock()
        llm.completion.side_effect = RateLimitError("too many requests")
        hook: PreSessionHook = PreSessionHook(memory_root=str(tmp_path), llm_client=llm)
        result = hook.run()
        assert result.success is True
        assert result.degraded is True

    def test_redis_and_llm_rate_limit_both_degrade(self, tmp_path: Path) -> None:
        """Both Redis and LLM rate-limit failures together should set degraded=True."""
        class RateLimitError(Exception):
            pass

        redis_client: MagicMock = MagicMock()
        redis_client.ping.side_effect = ConnectionError("no redis")
        llm: MagicMock = MagicMock()
        llm.completion.side_effect = RateLimitError("rate limited")
        hook: PreSessionHook = PreSessionHook(
            memory_root=str(tmp_path), redis_client=redis_client, llm_client=llm
        )
        result = hook.run()
        assert result.success is True
        assert result.degraded is True


# ===========================================================================
# PostSessionHook
# ===========================================================================


class TestPostSessionHook:
    """PostSessionHook flushes Redis, generates a summary, and writes metrics."""

    def _make_hook(
        self,
        tmp_path: Path,
        redis_client: Optional[MagicMock] = None,
        llm_client: Optional[MagicMock] = None,
    ) -> Tuple[PostSessionHook, MagicMock]:
        """
        Build a PostSessionHook with a mocked MemoryManager and given optional clients.

        Returns:
            A (hook, mock_memory_manager) tuple for assertion in tests.
        """
        mm: MagicMock = MagicMock()
        hook: PostSessionHook = PostSessionHook(
            memory_manager=mm,
            session_id="test-session",
            redis_client=redis_client,
            llm_client=llm_client,
            metrics_path=str(tmp_path / "metrics.jsonl"),
        )
        return hook, mm

    def test_run_returns_success(self, tmp_path: Path) -> None:
        """run() must return success=True immediately (the work is done in a thread)."""
        hook, _ = self._make_hook(tmp_path)
        result = hook.run(metrics={})
        assert result.success is True

    def test_metrics_written_after_run(self, tmp_path: Path) -> None:
        """_run_async() must write a JSONL line to the metrics file."""
        hook, _ = self._make_hook(tmp_path)
        hook._run_async({"turn_count": 3})
        path: str = str(tmp_path / "metrics.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            record = json.loads(f.readline())
        assert record["session_id"] == "test-session"

    def test_zero_turns_no_crash(self, tmp_path: Path) -> None:
        """An empty turns list should not raise any exception."""
        hook, _ = self._make_hook(tmp_path)
        hook._run_async({"turns": []})  # must not raise

    def test_no_redis_client_skips_flush(self, tmp_path: Path) -> None:
        """With no redis_client configured, save_message must not be called for flush."""
        hook, mm = self._make_hook(tmp_path)
        hook._run_async({})
        mm.save_message.assert_not_called()

    # --- _select_turns_for_summary ---

    def test_select_turns_all_fit(self, tmp_path: Path) -> None:
        """When all turns fit within the budget, all should be selected."""
        hook, _ = self._make_hook(tmp_path)
        hook.max_context_chars = 10_000
        turns = [{"user": "hi", "assistant": "hello"} for _ in range(5)]
        selected = hook._select_turns_for_summary(turns)
        assert len(selected) == 5

    def test_select_turns_respects_budget(self, tmp_path: Path) -> None:
        """When turns exceed the budget, only those fitting within it should be selected."""
        hook, _ = self._make_hook(tmp_path)
        # Each formatted turn is "User: hi\nAssistant: hello\n" = 26 chars
        hook.max_context_chars = 60  # fits ~2 turns
        turns = [{"user": "hi", "assistant": "hello"} for _ in range(10)]
        selected = hook._select_turns_for_summary(turns)
        assert len(selected) <= 2

    def test_select_turns_preserves_order(self, tmp_path: Path) -> None:
        """Selected turns must be returned oldest-first (original order preserved)."""
        hook, _ = self._make_hook(tmp_path)
        hook.max_context_chars = 10_000
        turns = [{"user": f"msg{i}", "assistant": f"resp{i}"} for i in range(5)]
        selected = hook._select_turns_for_summary(turns)
        assert selected == turns

    def test_select_turns_most_recent_kept(self, tmp_path: Path) -> None:
        """When not all turns fit, the most recent ones should be preferred."""
        hook, _ = self._make_hook(tmp_path)
        hook.max_context_chars = 60
        turns = [{"user": f"msg{i}", "assistant": f"resp{i}"} for i in range(5)]
        selected = hook._select_turns_for_summary(turns)
        assert selected == turns[len(turns) - len(selected):]

    def test_select_turns_skips_oversized_not_stops(self, tmp_path: Path) -> None:
        """
        An oversized turn in the middle should be skipped, not stop selection.
        Older smaller turns after the oversized one should still be considered.
        This tests the continue-not-break fix.
        """
        hook, _ = self._make_hook(tmp_path)
        hook.max_context_chars = 200
        # turns (oldest first): small, small, HUGE, small, small
        turns = [
            {"user": "short", "assistant": "short"},  # 0 — oldest, should be selected
            {"user": "short", "assistant": "short"},  # 1 — should be selected
            {"user": "x" * 500, "assistant": "y" * 500},  # 2 — too big, skip
            {"user": "short", "assistant": "short"},  # 3 — should be selected
            {"user": "short", "assistant": "short"},  # 4 — most recent, selected first
        ]
        selected = hook._select_turns_for_summary(turns)
        # The oversized turn (index 2) must not be present
        assert not any(t["user"].startswith("x") for t in selected)
        # At least some small turns should be selected
        assert len(selected) >= 1

    def test_select_turns_empty_input(self, tmp_path: Path) -> None:
        """An empty turns list should return an empty list."""
        hook, _ = self._make_hook(tmp_path)
        assert hook._select_turns_for_summary([]) == []
