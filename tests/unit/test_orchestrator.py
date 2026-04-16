"""
Unit tests for the Orchestrator module.

Covers:
  - _strip_thinking: CoT tag removal from raw LLM responses
  - _check_context_budget: alert fires only when input exceeds 90 % of max_context_chars
  - _call_agent_with_timeout: normal return path and TimeoutError on slow agents
  - _process_turn: per-turn metrics (hit/miss counters, turn_count, save failure handling)
"""

import time
import pytest
from typing import Callable
from unittest.mock import MagicMock, patch

from src.orchestrator import Orchestrator, SessionMetrics, _strip_thinking
from src.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator(agent: Callable = None, max_context_chars: int = 8000) -> Orchestrator:
    """
    Build a minimal Orchestrator with all dependencies mocked.

    Args:
        agent: Optional callable to use as the Strands agent. Defaults to a
               mock that returns "ok".
        max_context_chars: Value to set on the config for budget tests.

    Returns:
        A fully constructed Orchestrator with mocked hooks and memory_manager.
    """
    cfg = MagicMock(spec=Config)
    cfg.tool_timeout_seconds = 5
    cfg.max_tool_calls = 10
    cfg.max_context_chars = max_context_chars

    mm = MagicMock()
    mm.get_context_with_keys.return_value = ("", [])
    mm.eviction_count = 0

    pre_mem = MagicMock()
    pre_mem.run.return_value = MagicMock(success=True, message="query")

    post_mem = MagicMock()
    post_mem.run.return_value = MagicMock(success=True, message="")

    post_session = MagicMock()

    return Orchestrator(
        agent=agent or (lambda x: "ok"),
        memory_manager=mm,
        pre_mem_fetch_hook=pre_mem,
        post_mem_fetch_hook=post_mem,
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
        """Basic <think>…</think> block should be removed entirely."""
        raw: str = "<think>step 1\nstep 2</think>Hello!"
        assert _strip_thinking(raw) == "Hello!"

    def test_strips_thinking_tag(self) -> None:
        """<thinking>…</thinking> variant should also be removed."""
        raw: str = "<thinking>internal reasoning</thinking>Final answer."
        assert _strip_thinking(raw) == "Final answer."

    def test_case_insensitive(self) -> None:
        """Tag matching is case-insensitive (e.g. <THINK> should be stripped)."""
        raw: str = "<THINK>ignore</THINK>result"
        assert _strip_thinking(raw) == "result"

    def test_no_thinking_block_unchanged(self) -> None:
        """Responses without a CoT block must be returned unchanged."""
        raw: str = "Just a normal response."
        assert _strip_thinking(raw) == "Just a normal response."

    def test_strips_leading_whitespace_after_removal(self) -> None:
        """Whitespace left after tag removal should be stripped from the result."""
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
# _check_context_budget
# ---------------------------------------------------------------------------


class TestCheckContextBudget:
    """
    _check_context_budget emits an ERROR log when the input character count
    exceeds 90 % of max_context_chars.  It must NOT multiply by CHARS_PER_TOKEN.
    """

    def test_under_threshold_no_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """Input well below the threshold must not produce any log output."""
        orch = _make_orchestrator(max_context_chars=8000)
        with caplog.at_level("ERROR"):
            orch._check_context_budget("x" * 100)
        assert not caplog.records

    def test_at_threshold_alert_fires(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        Input at exactly 90 % of max_context_chars (7 200 chars for 8 000 limit)
        should trigger an ERROR log.  This confirms the budget is NOT multiplied
        by CHARS_PER_TOKEN (which would push the threshold to 28 800).
        """
        orch = _make_orchestrator(max_context_chars=8000)
        threshold_chars: int = int(8000 * 0.9) + 1  # just over 90 %
        with caplog.at_level("ERROR"):
            orch._check_context_budget("x" * threshold_chars)
        assert any("context" in r.message.lower() for r in caplog.records)

    def test_chars_per_token_not_applied(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        A 10 000-char input with max_context_chars=8 000 must fire the alert.
        If CHARS_PER_TOKEN were mistakenly applied (budget = 32 000), no alert
        would fire — this test would catch that regression.
        """
        orch = _make_orchestrator(max_context_chars=8000)
        with caplog.at_level("ERROR"):
            orch._check_context_budget("x" * 10_000)
        assert caplog.records, "Alert must fire for 10 000 chars with limit 8 000"


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
        """An agent that takes longer than tool_timeout_seconds must raise TimeoutError."""
        def slow_agent(x: str) -> str:
            """Simulate a hung agent."""
            time.sleep(10)
            return "never"

        orch = _make_orchestrator(agent=slow_agent)
        orch.config.tool_timeout_seconds = 0.1  # very short timeout

        with pytest.raises(TimeoutError):
            orch._call_agent_with_timeout("test")

    def test_thinking_tags_stripped(self) -> None:
        """CoT tags in the agent response must be stripped before returning."""
        orch = _make_orchestrator(agent=lambda x: "<think>cot</think>Clean answer")
        result: str = orch._call_agent_with_timeout("test")
        assert result == "Clean answer"
        assert "<think>" not in result


# ---------------------------------------------------------------------------
# _process_turn
# ---------------------------------------------------------------------------


class TestProcessTurn:
    """_process_turn executes the full per-turn pipeline with all dependencies mocked."""

    def test_memory_hit_increments_counter(self) -> None:
        """When get_context_with_keys returns context, memory_hits must increment."""
        orch = _make_orchestrator()
        orch.memory_manager.get_context_with_keys.return_value = ("some context", ["k1"])
        orch._process_turn("hello")
        assert orch.metrics.memory_hits == 1
        assert orch.metrics.memory_misses == 0

    def test_memory_miss_increments_counter(self) -> None:
        """When get_context_with_keys returns empty context, memory_misses must increment."""
        orch = _make_orchestrator()
        orch.memory_manager.get_context_with_keys.return_value = ("", [])
        orch._process_turn("hello")
        assert orch.metrics.memory_misses == 1
        assert orch.metrics.memory_hits == 0

    def test_turn_count_increments(self) -> None:
        """turn_count must increment by 1 for each successfully completed turn."""
        orch = _make_orchestrator()
        orch._process_turn("first")
        orch._process_turn("second")
        assert orch.metrics.turn_count == 2

    def test_save_failure_does_not_abort_turn(self) -> None:
        """
        A failure in save_message must be caught and logged — the turn response
        must still be returned so the user is not affected by a storage error.
        """
        orch = _make_orchestrator()
        orch.memory_manager.save_message.side_effect = OSError("disk full")
        result: str = orch._process_turn("hello")
        # Turn should complete and return the agent response despite save failure
        assert result is not None
        assert isinstance(result, str)
