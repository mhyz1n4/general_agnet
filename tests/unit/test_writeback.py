"""
Unit tests for tool-result episodic write-back and cache-read metric
propagation.

The write-back path saves one topic-tagged episodic entry per unique tool
invoked during a turn; tool names come from
``AgentStateContext.tools_invoked`` (populated by ``enforce_tool_budget``).
The metric path mirrors Strands' ``accumulated_usage.cacheReadInputTokens``
into ``SessionMetrics``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, List
from unittest.mock import MagicMock

from src.agents.context import AgentStateContext
from src.config import Config
from src.memory.stub_provider import StubMemoryProvider
from src.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator(agent: Callable, provider=None) -> Orchestrator:
    """Build an Orchestrator with a MagicMock Config and optional stub provider."""
    cfg = MagicMock(spec=Config)
    cfg.tool_timeout_seconds = 5
    cfg.max_tool_calls = 20
    cfg.max_context_chars = 8000
    cfg.tool_budget_web_search = 10
    cfg.tool_budget_run_python = 2
    cfg.tool_budget_delegate_to_research = 1
    cfg.memory_root = "./memory"

    return Orchestrator(
        agent=agent,
        memory_provider=provider or StubMemoryProvider(),
        post_session_hook=MagicMock(),
        config=cfg,
        session_id="test-session",
    )


class _AgentStub:
    """
    Callable agent stub that populates ``state_context.tools_invoked`` inline,
    mimicking what the ``enforce_tool_budget`` hook does at runtime.
    """

    def __init__(self, response: str, tool_names: List[str]) -> None:
        """Record the answer string and the tool names to report as invoked."""
        self.response = response
        self._tool_names = list(tool_names)
        # State context starts unset — Orchestrator attaches one on init.
        self.state_context: AgentStateContext | None = None

    def __call__(self, _input: str) -> str:
        """Simulate one agent loop: record tool invocations, return response."""
        if self.state_context is not None:
            self.state_context.tools_invoked.extend(self._tool_names)
            self.state_context.tool_call_count += len(self._tool_names)
        return self.response


# ---------------------------------------------------------------------------
# Write-back wired through _process_turn
# ---------------------------------------------------------------------------


def test_process_turn_saves_one_episodic_entry_per_tool() -> None:
    """Each unique tool name emits a separate topic-tagged episodic entry."""
    agent = _AgentStub("final answer", tool_names=["web_search", "search_memory"])
    orch = _make_orchestrator(agent=agent)

    orch._process_turn("what's new?")

    items = list(orch.memory_provider._items.values())
    topics = sorted(i.topic for i in items)
    assert "test-session" in topics
    assert "web_search" in topics
    assert "search_memory" in topics

    tool_entries = [i for i in items if i.topic == "web_search"]
    assert len(tool_entries) == 1
    assert "final answer" in tool_entries[0].content
    assert "tool=web_search" in tool_entries[0].content


def test_process_turn_dedupes_repeated_tool_calls() -> None:
    """Repeated uses of the same tool collapse to a single episodic entry."""
    agent = _AgentStub("ok", tool_names=["web_search", "web_search", "web_search"])
    orch = _make_orchestrator(agent=agent)

    orch._process_turn("go")

    items = list(orch.memory_provider._items.values())
    web_entries = [i for i in items if i.topic == "web_search"]
    assert len(web_entries) == 1


def test_process_turn_no_tools_writes_only_session_entry() -> None:
    """A turn with zero tool calls emits only the session-topic entry."""
    agent = _AgentStub("hi", tool_names=[])
    orch = _make_orchestrator(agent=agent)

    orch._process_turn("hello")

    items = list(orch.memory_provider._items.values())
    assert len(items) == 1
    assert items[0].topic == "test-session"


def test_write_back_failure_does_not_abort_turn() -> None:
    """If a per-tool save raises, the turn still completes."""
    provider = StubMemoryProvider()
    raises = iter([False, True, False])   # second save raises

    original_save = provider.save

    def _flaky_save(*args, **kwargs):
        """Wrap ``provider.save`` so the second invocation raises once."""
        if next(raises):
            raise RuntimeError("disk error")
        return original_save(*args, **kwargs)

    provider.save = _flaky_save  # type: ignore[assignment]

    agent = _AgentStub("ok", tool_names=["web_search", "search_memory"])
    orch = _make_orchestrator(agent=agent, provider=provider)

    response = orch._process_turn("go")
    assert response == "ok"
    assert orch.metrics.turn_count == 1


# ---------------------------------------------------------------------------
# cache_read_input_tokens metric
# ---------------------------------------------------------------------------


def test_cache_read_input_tokens_propagates_from_strands() -> None:
    """When Strands exposes cacheReadInputTokens, it surfaces on SessionMetrics."""
    class _Agent:
        """Callable agent that also exposes Strands-style metrics."""

        def __init__(self) -> None:
            """Attach an ``accumulated_usage`` payload with a cache counter."""
            self.event_loop_metrics = SimpleNamespace(
                accumulated_usage={
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "totalTokens": 150,
                    "cacheReadInputTokens": 42,
                }
            )

        def __call__(self, _input: str) -> str:
            """Return a constant response; the metric is read from the agent."""
            return "done"

    orch = _make_orchestrator(agent=_Agent())
    orch._call_agent_with_timeout("hello")
    assert orch.metrics.cache_read_input_tokens == 42


def test_cache_read_input_tokens_defaults_to_zero() -> None:
    """Agents without Strands-style metrics leave the counter at 0."""
    orch = _make_orchestrator(agent=lambda _i: "done")
    orch._call_agent_with_timeout("hello")
    assert orch.metrics.cache_read_input_tokens == 0
