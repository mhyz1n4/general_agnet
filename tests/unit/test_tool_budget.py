"""
Unit tests for the per-loop tool-call budget.

Budget enforcement lives in two layers:
  - ``AgentStateContext`` (``src/agents/context.py``) owns the counters and
    returns a cancel-reason string when a limit is exceeded.
  - ``enforce_tool_budget`` (``src/hooks/tool_budget.py``) is a stateless
    ``BeforeToolCallEvent`` callback that reads the context off
    ``event.agent`` and populates ``event.cancel_tool`` on overflow.

Orchestrator-level wiring (state context attached on construction, reset
per turn) is covered too.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock

from src.agents.context import AgentStateContext
from src.config import Config
from src.hooks.tool_budget import enforce_tool_budget
from src.memory.stub_provider import StubMemoryProvider
from src.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator(agent: Callable) -> Orchestrator:
    """Build an Orchestrator with a MagicMock Config honouring per-tool fields."""
    cfg = MagicMock(spec=Config)
    cfg.agent_turn_timeout_seconds = 5
    cfg.max_tool_calls = 20
    cfg.max_context_chars = 8000
    cfg.tool_budget_web_search = 2
    cfg.tool_budget_run_python = 1
    cfg.tool_budget_delegate_to_research = 1

    return Orchestrator(
        agent=agent,
        memory_provider=StubMemoryProvider(),
        post_session_hook=MagicMock(),
        config=cfg,
        session_id="test",
    )


def _fake_event(agent: object, tool_name: str) -> SimpleNamespace:
    """Construct a ``BeforeToolCallEvent``-shaped stand-in for the hook."""
    return SimpleNamespace(
        agent=agent,
        tool_use={"name": tool_name},
        cancel_tool=False,
    )


# ---------------------------------------------------------------------------
# AgentStateContext — pure counter logic
# ---------------------------------------------------------------------------


class TestAgentStateContext:
    """``check_and_increment`` / ``reset`` form the core of the budget logic."""

    def test_unlimited_when_no_caps_configured(self) -> None:
        """Absent limits mean every tool call is permitted."""
        ctx = AgentStateContext()
        for _ in range(100):
            assert ctx.check_and_increment("web_search") is None

    def test_global_cap_enforced(self) -> None:
        """Once ``max_tool_calls`` is exceeded every further call is cancelled."""
        ctx = AgentStateContext(max_tool_calls=2)
        assert ctx.check_and_increment("foo") is None
        assert ctx.check_and_increment("foo") is None
        reason = ctx.check_and_increment("foo")
        assert reason is not None and "3/2" in reason

    def test_per_tool_cap_isolated_from_other_tools(self) -> None:
        """A per-tool cap does not block unrelated tools."""
        ctx = AgentStateContext(
            max_tool_calls=10, per_tool_limits={"web_search": 1}
        )
        assert ctx.check_and_increment("web_search") is None
        reason = ctx.check_and_increment("web_search")
        assert reason is not None and "web_search" in reason and "2/1" in reason
        # A different tool still has budget.
        assert ctx.check_and_increment("memorize") is None

    def test_reset_zeroes_counters_but_keeps_limits(self) -> None:
        """``reset`` restores counters while preserving configured caps."""
        ctx = AgentStateContext(
            max_tool_calls=2, per_tool_limits={"web_search": 1}
        )
        ctx.check_and_increment("web_search")
        ctx.check_and_increment("web_search")
        assert ctx.tool_call_count == 2
        ctx.reset()
        assert ctx.tool_call_count == 0
        assert ctx.per_tool_counts == {}
        # Caps still enforced after reset.
        assert ctx.check_and_increment("web_search") is None

    def test_unique_tools_invoked_preserves_order(self) -> None:
        """``unique_tools_invoked`` dedupes while preserving first-seen order."""
        ctx = AgentStateContext()
        ctx.tools_invoked = ["a", "b", "a", "c", "b"]
        assert ctx.unique_tools_invoked() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# enforce_tool_budget — Strands hook callback
# ---------------------------------------------------------------------------


class TestEnforceToolBudgetHook:
    """``enforce_tool_budget`` is a pure function of the event + agent state."""

    def test_agent_without_context_is_unrestricted(self) -> None:
        """Agents without ``state_context`` must not be cancelled."""
        agent = SimpleNamespace()   # no state_context attribute
        event = _fake_event(agent, "web_search")
        enforce_tool_budget(event)
        assert event.cancel_tool is False

    def test_cancels_when_per_tool_budget_exceeded(self) -> None:
        """Tool cap exceeded => event.cancel_tool set to the reason string."""
        ctx = AgentStateContext(per_tool_limits={"web_search": 1})
        agent = SimpleNamespace(state_context=ctx)
        enforce_tool_budget(_fake_event(agent, "web_search"))
        event = _fake_event(agent, "web_search")
        enforce_tool_budget(event)
        assert isinstance(event.cancel_tool, str)
        assert "web_search" in event.cancel_tool

    def test_successful_call_is_tracked_for_writeback(self) -> None:
        """Allowed calls feed ``tools_invoked`` so the orchestrator can write back."""
        ctx = AgentStateContext()
        agent = SimpleNamespace(state_context=ctx)
        enforce_tool_budget(_fake_event(agent, "search_memory"))
        enforce_tool_budget(_fake_event(agent, "web_search"))
        assert ctx.tools_invoked == ["search_memory", "web_search"]

    def test_global_cap_cancels_regardless_of_tool(self) -> None:
        """The global cap applies even to unbudgeted tools."""
        ctx = AgentStateContext(max_tool_calls=1)
        agent = SimpleNamespace(state_context=ctx)
        enforce_tool_budget(_fake_event(agent, "memorize"))
        event = _fake_event(agent, "memorize")
        enforce_tool_budget(event)
        assert isinstance(event.cancel_tool, str)
        assert "2/1" in event.cancel_tool


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------


class TestOrchestratorAttachesStateContext:
    """The Orchestrator attaches a state context with the config's budgets."""

    def test_state_context_created_with_expected_limits(self) -> None:
        """A fresh orchestrator must install an ``AgentStateContext``."""
        orch = _make_orchestrator(agent=lambda _i: "done")
        ctx = getattr(orch.agent, "state_context", None)
        assert isinstance(ctx, AgentStateContext)
        assert ctx.max_tool_calls == 20
        assert ctx.per_tool_limits == {
            "web_search": 2,
            "run_python": 1,
            "delegate_to_research": 1,
        }

    def test_state_context_reset_each_turn(self) -> None:
        """``_call_agent_with_timeout`` resets counters before the agent runs."""
        captured: dict[str, int] = {}

        def _agent(_input: str) -> str:
            """Snapshot the counter state that the orchestrator set up."""
            ctx = agent_holder["agent"].state_context
            captured["count_at_start"] = ctx.tool_call_count
            ctx.check_and_increment("web_search")
            return "done"

        orch = _make_orchestrator(agent=_agent)
        agent_holder = {"agent": orch.agent}

        # Preload counts from a previous turn — must be wiped on next call.
        orch.agent.state_context.tool_call_count = 7
        orch._call_agent_with_timeout("x")
        assert captured["count_at_start"] == 0

    def test_preexisting_state_context_is_preserved(self) -> None:
        """A caller who attached their own context must not have it overwritten."""
        ctx = AgentStateContext(max_tool_calls=999)

        class _Agent:
            """Callable agent fixture with a pre-attached state context."""

            state_context = ctx

            def __call__(self, _input: str) -> str:
                """Return a constant response; not exercised here."""
                return "done"

        my_agent = _Agent()
        cfg = MagicMock(spec=Config)
        cfg.agent_turn_timeout_seconds = 5
        cfg.max_tool_calls = 3
        cfg.max_context_chars = 8000
        cfg.tool_budget_web_search = 10

        orch = Orchestrator(
            agent=my_agent,
            memory_provider=StubMemoryProvider(),
            post_session_hook=MagicMock(),
            config=cfg,
            session_id="test",
        )
        assert orch.agent.state_context is ctx
        assert ctx.max_tool_calls == 999
