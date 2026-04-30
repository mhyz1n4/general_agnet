"""
Unit tests for tool-result episodic write-back and cache-read metric
propagation.

The write-back path saves one topic-tagged episodic entry per unique tool
invoked during a turn; tool names come from
``AgentStateContext.tools_invoked`` (populated by ``enforce_tool_budget``).
The metric path mirrors Strands' ``accumulated_usage.cacheReadInputTokens``
into ``SessionMetrics``.

Compaction + archive (issue c): episodic entries hold a *truncated* turn
summary (cap = ``TURN_SUMMARY_MAX_CHARS``); the full transcript is appended
to ``<memory_root>/archive/session_<id>.jsonl``.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Callable, List
from unittest.mock import MagicMock, patch

from src.agents.context import AgentStateContext
from src.config import Config
from src.constants import TURN_SUMMARY_MAX_CHARS
from src.memory.stub_provider import StubMemoryProvider
from src.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator(
    agent: Callable,
    provider=None,
    memory_root: str = "./memory",
) -> Orchestrator:
    """Build an Orchestrator with a MagicMock Config and optional stub provider."""
    cfg = MagicMock(spec=Config)
    cfg.agent_turn_timeout_seconds = 5
    cfg.max_tool_calls = 20
    cfg.max_context_chars = 8000
    cfg.tool_budget_web_search = 10
    cfg.tool_budget_run_python = 2
    cfg.tool_budget_delegate_to_research = 1
    cfg.memory_root = memory_root

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


def test_process_turn_saves_one_episodic_entry_per_tool(tmp_path) -> None:
    """Each unique tool name emits a separate topic-tagged episodic entry."""
    agent = _AgentStub("final answer", tool_names=["web_search", "search_memory"])
    orch = _make_orchestrator(agent=agent, memory_root=str(tmp_path))

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


def test_process_turn_dedupes_repeated_tool_calls(tmp_path) -> None:
    """Repeated uses of the same tool collapse to a single episodic entry."""
    agent = _AgentStub("ok", tool_names=["web_search", "web_search", "web_search"])
    orch = _make_orchestrator(agent=agent, memory_root=str(tmp_path))

    orch._process_turn("go")

    items = list(orch.memory_provider._items.values())
    web_entries = [i for i in items if i.topic == "web_search"]
    assert len(web_entries) == 1


def test_process_turn_no_tools_writes_only_session_entry(tmp_path) -> None:
    """A turn with zero tool calls emits only the session-topic entry."""
    agent = _AgentStub("hi", tool_names=[])
    orch = _make_orchestrator(agent=agent, memory_root=str(tmp_path))

    orch._process_turn("hello")

    items = list(orch.memory_provider._items.values())
    assert len(items) == 1
    assert items[0].topic == "test-session"


def test_write_back_failure_does_not_abort_turn(tmp_path) -> None:
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
    orch = _make_orchestrator(agent=agent, provider=provider, memory_root=str(tmp_path))

    response = orch._process_turn("go")
    assert response == "ok"
    assert orch.metrics.turn_count == 1


# ---------------------------------------------------------------------------
# Compaction + raw archive (issue c)
# ---------------------------------------------------------------------------


def _archive_lines(memory_root: str, session_id: str = "test-session") -> List[dict]:
    """Read the per-session archive jsonl and return parsed records."""
    path = os.path.join(memory_root, "archive", f"session_{session_id}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_archive_appends_one_record_per_turn(tmp_path) -> None:
    """Each turn appends a JSON line containing the full user/assistant text."""
    agent = _AgentStub("hi there", tool_names=[])
    orch = _make_orchestrator(agent=agent, memory_root=str(tmp_path))

    orch._process_turn("hello")

    records = _archive_lines(str(tmp_path))
    assert len(records) == 1
    assert records[0]["user"] == "hello"
    assert records[0]["assistant"] == "hi there"
    assert records[0]["session_id"] == "test-session"
    assert records[0]["tools_invoked"] == []


def test_archive_appends_multiple_turns(tmp_path) -> None:
    """Successive turns are appended without overwriting prior records."""
    agent = _AgentStub("a1", tool_names=["web_search"])
    orch = _make_orchestrator(agent=agent, memory_root=str(tmp_path))

    orch._process_turn("q1")
    agent.response = "a2"
    orch._process_turn("q2")

    records = _archive_lines(str(tmp_path))
    assert [r["user"] for r in records] == ["q1", "q2"]
    assert [r["assistant"] for r in records] == ["a1", "a2"]
    assert records[0]["tools_invoked"] == ["web_search"]


def test_archive_holds_full_response_episodic_truncates(tmp_path) -> None:
    """Long responses are stored verbatim in the archive but clipped in episodic memory."""
    long_response = "x" * (TURN_SUMMARY_MAX_CHARS * 4)
    agent = _AgentStub(long_response, tool_names=["web_search"])
    orch = _make_orchestrator(agent=agent, memory_root=str(tmp_path))

    orch._process_turn("go")

    # Archive carries the untruncated assistant text.
    records = _archive_lines(str(tmp_path))
    assert records[0]["assistant"] == long_response

    # Episodic entries are clipped to TURN_SUMMARY_MAX_CHARS per half.
    items = list(orch.memory_provider._items.values())
    tool_entry = next(i for i in items if i.topic == "web_search")
    payload = tool_entry.content.split("summary: ", 1)[1]
    assert len(payload) == TURN_SUMMARY_MAX_CHARS
    assert payload == "x" * TURN_SUMMARY_MAX_CHARS

    session_entry = next(i for i in items if i.topic == "test-session")
    assert ("x" * TURN_SUMMARY_MAX_CHARS) in session_entry.content
    assert ("x" * (TURN_SUMMARY_MAX_CHARS + 1)) not in session_entry.content


def test_archive_failure_does_not_abort_turn(tmp_path) -> None:
    """An OSError while writing the archive is logged and swallowed."""
    agent = _AgentStub("ok", tool_names=[])
    orch = _make_orchestrator(agent=agent, memory_root=str(tmp_path))

    with patch("builtins.open", side_effect=OSError("disk full")):
        response = orch._process_turn("go")

    assert response == "ok"
    assert orch.metrics.turn_count == 1


def test_archive_disabled_when_memory_root_unset() -> None:
    """When ``memory_root`` is non-string (e.g. MagicMock), no archive is written."""
    cfg = MagicMock(spec=Config)
    cfg.agent_turn_timeout_seconds = 5
    cfg.max_tool_calls = 20
    cfg.max_context_chars = 8000
    cfg.tool_budget_web_search = 10
    cfg.tool_budget_run_python = 2
    cfg.tool_budget_delegate_to_research = 1
    # memory_root deliberately left as MagicMock attribute (not a string).

    orch = Orchestrator(
        agent=_AgentStub("ok", tool_names=[]),
        memory_provider=StubMemoryProvider(),
        post_session_hook=MagicMock(),
        config=cfg,
        session_id="test-session",
    )
    assert orch._archive_path is None
    # The turn must not raise even though the archive is disabled.
    assert orch._process_turn("go") == "ok"


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
