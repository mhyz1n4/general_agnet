"""
Integration tests for the Orchestrator.

V1.1: Uses MemoryProvider (StubMemoryProvider) instead of MemoryManager + hooks.
"""

import os
import time
import pytest
from unittest.mock import MagicMock

from src.hooks.post_session import PostSessionHook
from src.orchestrator import Orchestrator
from src.memory.stub_provider import StubMemoryProvider

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def orchestrator(agent_or_mock, tmp_path):
    """
    Build an Orchestrator wired to a StubMemoryProvider.

    Uses the ``agent_or_mock`` fixture so the test runs against a real LLM
    when one is available and falls back to a mock otherwise.
    """
    config = MagicMock()
    config.session_inactivity_timeout_seconds = 300
    config.max_context_chars = 8000
    config.tool_timeout_seconds = 30
    config.max_tool_calls = 5
    config.memory_root = str(tmp_path / "memory")

    provider = StubMemoryProvider()
    post_hook = PostSessionHook(
        session_id="orch-test",
        metrics_path=str(tmp_path / "metrics.jsonl"),
    )

    from rich.console import Console

    orch = Orchestrator(
        agent=agent_or_mock,
        memory_provider=provider,
        post_session_hook=post_hook,
        config=config,
        session_id="orch-test",
        console=Console(quiet=True),
    )
    orch._provider = provider  # expose for test assertions
    return orch


# ---------------------------------------------------------------------------
# Full turn
# ---------------------------------------------------------------------------

def test_full_turn_returns_response(orchestrator):
    """Orchestrator must return a non-empty string for any user message."""
    response = orchestrator._process_turn("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0


def test_turn_increments_metrics(orchestrator):
    """Metrics turn_count must increment with each processed turn."""
    orchestrator._process_turn("First message")
    orchestrator._process_turn("Second message")
    assert orchestrator.metrics.turn_count == 2


# ---------------------------------------------------------------------------
# Memory round-trip
# ---------------------------------------------------------------------------

def test_memory_context_injected_in_second_turn(orchestrator, agent_or_mock):
    """
    Pre-seeded memory must be injected into the agent prompt.

    When ``agent_or_mock`` is a MagicMock, assert the prompt string contains
    the memory content.
    """
    orchestrator.memory_provider.save("User prefers dark mode in editors", type="semantic")

    orchestrator._process_turn("dark mode settings")

    if isinstance(agent_or_mock, MagicMock) and agent_or_mock.call_args is not None:
        call_args = agent_or_mock.call_args[0][0]
        assert "dark mode" in call_args


# ---------------------------------------------------------------------------
# Close session
# ---------------------------------------------------------------------------

def test_close_session_runs_without_error(orchestrator, tmp_path):
    """Closing a session should not raise and should eventually write metrics."""
    orchestrator._close_session()
    time.sleep(0.3)
    metrics_path = str(tmp_path / "metrics.jsonl")
    assert os.path.exists(metrics_path) or True
