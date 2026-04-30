"""
Unit tests for the Orchestrator module.

Covers:
  - _strip_thinking: CoT tag removal from raw LLM responses
  - _normalize_query: strip punctuation, lowercase, collapse whitespace for memory search
  - _check_context_budget: alert fires only when input exceeds 90% of max_context_chars
  - _call_agent_with_timeout: normal return path and TimeoutError on slow agents
  - _process_turn: per-turn metrics (hit/miss counters, turn_count, save failure handling)
"""

import time
import pytest
from typing import Callable
from unittest.mock import MagicMock, patch

from src.orchestrator import (
    Orchestrator,
    SessionMetrics,
    _build_context_input,
    _normalize_query,
    _strip_thinking,
)
from src.config import Config
from src.memory.provider import MemoryItem
from src.memory.stub_provider import StubMemoryProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator(agent: Callable = None, max_context_chars: int = 8000) -> Orchestrator:
    """
    Build a minimal Orchestrator with dependencies mocked or stubbed.

    Args:
        agent: Optional callable to use as the Strands agent.
        max_context_chars: Value to set on the config for budget tests.

    Returns:
        A fully constructed Orchestrator with StubMemoryProvider and mocked hooks.
    """
    cfg = MagicMock(spec=Config)
    cfg.agent_turn_timeout_seconds = 5
    cfg.max_tool_calls = 10
    cfg.max_context_chars = max_context_chars

    memory_provider = StubMemoryProvider()
    post_session = MagicMock()

    return Orchestrator(
        agent=agent or (lambda x: "ok"),
        memory_provider=memory_provider,
        post_session_hook=post_session,
        config=cfg,
        session_id="test-session",
    )


# ---------------------------------------------------------------------------
# _strip_thinking
# ---------------------------------------------------------------------------


class TestStripThinking:
    """_strip_thinking removes <think> / <thinking> CoT blocks from LLM output."""

    def test_strips_think_tag(self) -> None:
        """Basic <think>...</think> block should be removed entirely."""
        raw: str = "<think>step 1\nstep 2</think>Hello!"
        assert _strip_thinking(raw) == "Hello!"

    def test_strips_thinking_tag(self) -> None:
        """<thinking>...</thinking> variant should also be removed."""
        raw: str = "<thinking>internal reasoning</thinking>Final answer."
        assert _strip_thinking(raw) == "Final answer."

    def test_case_insensitive(self) -> None:
        """Tag matching is case-insensitive."""
        raw: str = "<THINK>ignore</THINK>result"
        assert _strip_thinking(raw) == "result"

    def test_no_thinking_block_unchanged(self) -> None:
        """Responses without a CoT block must be returned unchanged."""
        raw: str = "Just a normal response."
        assert _strip_thinking(raw) == "Just a normal response."

    def test_strips_leading_whitespace_after_removal(self) -> None:
        """Whitespace left after tag removal should be stripped."""
        raw: str = "<think>cot</think>\n\nActual response."
        assert _strip_thinking(raw) == "Actual response."

    def test_empty_string(self) -> None:
        """Empty input must return an empty string without raising."""
        assert _strip_thinking("") == ""

    def test_only_thinking_block(self) -> None:
        """A response that is entirely a CoT block should return an empty string."""
        raw: str = "<think>only cot, no visible output</think>"
        assert _strip_thinking(raw) == ""

    def test_multiline_thinking_block(self) -> None:
        """Multi-line CoT blocks must be stripped in their entirety."""
        raw: str = "<think>\nline 1\nline 2\nline 3\n</think>Answer here."
        assert _strip_thinking(raw) == "Answer here."


# ---------------------------------------------------------------------------
# _normalize_query
# ---------------------------------------------------------------------------


class TestNormalizeQuery:
    """_normalize_query strips punctuation, lowercases, and collapses whitespace."""

    def test_empty_string(self) -> None:
        """An empty query should produce an empty string."""
        assert _normalize_query("") == ""

    def test_special_chars_stripped(self) -> None:
        """Non-alphanumeric characters (except spaces/apostrophes) should be removed."""
        assert _normalize_query("!@#$%") == ""

    def test_apostrophe_kept(self) -> None:
        """Apostrophes in contractions should be preserved."""
        assert _normalize_query("what's up") == "what's up"

    def test_lowercased(self) -> None:
        """All characters should be converted to lowercase."""
        assert _normalize_query("Hello World") == "hello world"

    def test_collapse_whitespace(self) -> None:
        """Multiple consecutive spaces should collapse to a single space."""
        assert _normalize_query("hello  world") == "hello world"


# ---------------------------------------------------------------------------
# _build_context_input
# ---------------------------------------------------------------------------


def _item(content: str) -> MemoryItem:
    """Build a minimal MemoryItem for context-budget tests."""
    return MemoryItem(key="k", content=content, type="semantic")


class TestBuildContextInput:
    """_build_context_input must trim tail items so the rendered input fits the budget."""

    def test_no_results_returns_user_input(self) -> None:
        """Empty result list yields full_input that contains the user query and zero drops."""
        full_input, dropped = _build_context_input([], "hello", max_chars=10_000)
        assert "hello" in full_input
        assert dropped == 0

    def test_all_items_fit(self) -> None:
        """When everything fits, nothing is dropped."""
        results = [_item("a"), _item("b")]
        full_input, dropped = _build_context_input(results, "hi", max_chars=10_000)
        assert dropped == 0
        assert "a" in full_input and "b" in full_input

    def test_tail_items_dropped(self) -> None:
        """Items added in order; the tail is dropped once the budget is hit."""
        results = [_item("AAAA"), _item("BBBB"), _item("CCCC")]
        baseline, _ = _build_context_input([results[0]], "q", max_chars=10_000)
        budget = len(baseline) + 1  # only the first item should fit
        _, dropped = _build_context_input(results, "q", max_chars=budget)
        assert dropped == 2

    def test_zero_budget_drops_everything(self) -> None:
        """A budget too small for any item still returns the user input verbatim."""
        results = [_item("X")]
        full_input, dropped = _build_context_input(results, "q", max_chars=1)
        assert dropped == 1
        assert "q" in full_input


# ---------------------------------------------------------------------------
# _check_context_budget
# ---------------------------------------------------------------------------


class TestCheckContextBudget:
    """_check_context_budget emits an ERROR log when input exceeds 90% of max_context_chars."""

    def test_under_threshold_no_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """Input well below the threshold must not produce any log output."""
        orch = _make_orchestrator(max_context_chars=8000)
        with caplog.at_level("ERROR"):
            orch._check_context_budget("x" * 100)
        assert not caplog.records

    def test_at_threshold_alert_fires(self, caplog: pytest.LogCaptureFixture) -> None:
        """Input at exactly 90% of max_context_chars should trigger an ERROR log."""
        orch = _make_orchestrator(max_context_chars=8000)
        threshold_chars: int = int(8000 * 0.9) + 1
        with caplog.at_level("ERROR"):
            orch._check_context_budget("x" * threshold_chars)
        assert any("context" in r.message.lower() for r in caplog.records)

    def test_chars_per_token_not_applied(self, caplog: pytest.LogCaptureFixture) -> None:
        """A 10000-char input with max_context_chars=8000 must fire the alert."""
        orch = _make_orchestrator(max_context_chars=8000)
        with caplog.at_level("ERROR"):
            orch._check_context_budget("x" * 10_000)
        assert caplog.records, "Alert must fire for 10000 chars with limit 8000"


# ---------------------------------------------------------------------------
# _call_agent_with_timeout
# ---------------------------------------------------------------------------


class TestCallAgentWithTimeout:
    """_call_agent_with_timeout wraps agent() with a wall-clock deadline."""

    def test_normal_call_returns_response(self) -> None:
        """A fast agent should return its response string."""
        orch = _make_orchestrator(agent=lambda x: "hello world")
        result: str = orch._call_agent_with_timeout("test input")
        assert "hello world" in result

    def test_timeout_raises_timeout_error(self) -> None:
        """An agent that exceeds agent_turn_timeout_seconds must raise TimeoutError."""
        def slow_agent(x: str) -> str:
            """Simulate a hung agent."""
            time.sleep(10)
            return "never"

        orch = _make_orchestrator(agent=slow_agent)
        orch.config.agent_turn_timeout_seconds = 0.1

        with pytest.raises(TimeoutError):
            orch._call_agent_with_timeout("test")

    def test_thinking_tags_stripped(self) -> None:
        """CoT tags in the agent response must be stripped before returning."""
        orch = _make_orchestrator(agent=lambda x: "<think>cot</think>Clean answer")
        result: str = orch._call_agent_with_timeout("test")
        assert result == "Clean answer"
        assert "<think>" not in result

    def test_cache_hits_propagate_to_metrics(self) -> None:
        """Cache hits recorded during an agent turn land on ``SessionMetrics``."""
        from src.tools.cache import cached_tool, get_session_cache
        from src.tools.envelope import ok

        @cached_tool("stub_tool")
        def _stub(x: int) -> dict:
            """Tool stub used to drive cache insertion from within an agent call."""
            return ok({"x": x})

        def _agent(_input: str) -> str:
            """Agent that calls the cached stub twice — second call should hit."""
            _stub(x=1)
            _stub(x=1)
            return "done"

        orch = _make_orchestrator(agent=_agent)
        orch._call_agent_with_timeout("test")
        assert orch.metrics.cache_hits == 1


# ---------------------------------------------------------------------------
# _process_turn
# ---------------------------------------------------------------------------


class TestProcessTurn:
    """_process_turn executes the full per-turn pipeline with MemoryProvider."""

    def test_memory_hit_increments_counter(self) -> None:
        """When search returns results, memory_hits must increment."""
        orch = _make_orchestrator()
        # Pre-seed the stub provider with searchable content
        orch.memory_provider.save("hello world facts", type="semantic")
        orch._process_turn("hello")
        assert orch.metrics.memory_hits == 1
        assert orch.metrics.memory_misses == 0

    def test_memory_miss_increments_counter(self) -> None:
        """When search returns no results, memory_misses must increment."""
        orch = _make_orchestrator()
        orch._process_turn("xyznonexistent")
        assert orch.metrics.memory_misses == 1
        assert orch.metrics.memory_hits == 0

    def test_turn_count_increments(self) -> None:
        """turn_count must increment by 1 for each successfully completed turn."""
        orch = _make_orchestrator()
        orch._process_turn("first")
        orch._process_turn("second")
        assert orch.metrics.turn_count == 2


# ---------------------------------------------------------------------------
# _close_session
# ---------------------------------------------------------------------------


class TestCloseSession:
    """_close_session must flush the conversation manager before writing metrics."""

    def test_flush_invoked_before_post_session_hook(self) -> None:
        """Conversation manager's flush() runs before the post-session hook."""
        agent = MagicMock()
        agent.conversation_manager = MagicMock()
        agent.messages = []

        orch = _make_orchestrator(agent=agent)
        orch.config.memory_root = "/nonexistent"

        call_order: list[str] = []
        agent.conversation_manager.flush.side_effect = lambda a: call_order.append("flush")
        orch.post_session_hook.run.side_effect = lambda **kw: call_order.append("post")

        orch._close_session()

        agent.conversation_manager.flush.assert_called_once_with(agent)
        assert call_order == ["flush", "post"]

    def test_missing_conversation_manager_tolerated(self) -> None:
        """A plain-callable agent (no conversation_manager) must not crash close."""
        orch = _make_orchestrator(agent=lambda x: "ok")
        orch.config.memory_root = "/nonexistent"

        orch._close_session()

        orch.post_session_hook.run.assert_called_once()

    def test_flush_exception_does_not_block_metrics(self) -> None:
        """A raising flush() must be logged and metrics must still be written."""
        agent = MagicMock()
        agent.conversation_manager = MagicMock()
        agent.conversation_manager.flush.side_effect = RuntimeError("boom")

        orch = _make_orchestrator(agent=agent)
        orch.config.memory_root = "/nonexistent"

        orch._close_session()

        orch.post_session_hook.run.assert_called_once()
