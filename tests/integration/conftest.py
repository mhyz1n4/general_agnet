"""
Shared fixtures and test data for integration tests.

V1.1: Memory stack uses ``StubMemoryProvider`` for fast isolated tests.
``ReMeLightProvider`` is available for integration tests that need it.

Fixtures
--------
mem_root     — isolated tmp directory for each test's memory files
memory_provider — StubMemoryProvider instance
session_id   — unique session ID per test
agent_or_mock — real Strands Agent when LLM is reachable, else a MagicMock

Tool-budget enforcement is wired via ``ToolBudgetHookProvider`` +
``AgentStateContext`` (see ``src/hooks/tool_budget.py`` and
``src/agents/context.py``).  Integration tests that drive an Agent directly
attach their own ``AgentStateContext`` when they need budget enforcement.

Test data
---------
TEST_MEMORIES — list of canonical memory records shared across integration tests.
"""

import os
import uuid
from unittest.mock import MagicMock

import pytest

from src.constants import SESSION_ID_HEX_LENGTH
from src.memory.stub_provider import StubMemoryProvider


def pytest_configure(config):
    """Register the ``integration`` marker so pytest does not warn about it."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

TEST_MEMORIES = [
    {
        "content": "Python is a programming language",
        "type": "semantic",
        "topic": "programming",
    },
    {
        "content": "I like coffee in the morning",
        "type": "semantic",
        "topic": "personal",
    },
    {
        "content": "Use pytest for Python testing",
        "type": "procedural",
        "topic": "testing",
    },
    {
        "content": "Today we had a team meeting",
        "type": "episodic",
        "topic": None,
    },
    {
        "content": "User prefers dark mode in editors",
        "type": "semantic",
        "topic": "preferences",
    },
]


# ---------------------------------------------------------------------------
# Test data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def TEST_MEMORIES():
    """Return the shared test memory records as a fixture."""
    return [
        {
            "content": "Python is a programming language",
            "type": "semantic",
            "topic": "programming",
        },
        {
            "content": "I like coffee in the morning",
            "type": "semantic",
            "topic": "personal",
        },
        {
            "content": "Use pytest for Python testing",
            "type": "procedural",
            "topic": "testing",
        },
        {
            "content": "Today we had a team meeting",
            "type": "episodic",
            "topic": None,
        },
        {
            "content": "User prefers dark mode in editors",
            "type": "semantic",
            "topic": "preferences",
        },
    ]


# ---------------------------------------------------------------------------
# Core memory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_root(tmp_path):
    """Isolated filesystem root for a single test's memory files."""
    return str(tmp_path / "memory")


@pytest.fixture
def memory_provider():
    """Fresh StubMemoryProvider for each test."""
    return StubMemoryProvider()


@pytest.fixture
def session_id() -> str:
    """Unique session ID per test run."""
    return f"test_{uuid.uuid4().hex[:SESSION_ID_HEX_LENGTH]}"


# ---------------------------------------------------------------------------
# Agent fixture — real or mock depending on LLM availability
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_or_mock(llm_endpoint_or_none):
    """
    Return a real Strands Agent when the LLM endpoint is reachable, else a
    ``MagicMock`` that returns a fixed response string.

    The Agent is built with ``ToolBudgetHookProvider`` so any attached
    ``AgentStateContext`` participates in per-turn budget enforcement.
    """
    if llm_endpoint_or_none is None:
        return MagicMock(return_value="This is the assistant response.")

    try:
        from strands import Agent
        from strands.models.openai import OpenAIModel
    except ImportError:
        return MagicMock(return_value="This is the assistant response.")

    from src.agents.context import AgentStateContext
    from src.config import Config
    from src.hooks.tool_budget import ToolBudgetHookProvider

    config = Config()

    model = OpenAIModel(
        client_args={
            "api_key": config.llm_api_key,
            "base_url": config.llm_api_endpoint,
        },
        model_id=config.llm_model,
        params={"max_tokens": max(config.llm_max_tokens, 4096)},
    )
    agent = Agent(
        model=model,
        tools=[],
        callback_handler=None,
        hooks=[ToolBudgetHookProvider()],
    )
    agent.state_context = AgentStateContext(max_tool_calls=config.max_tool_calls)
    return agent
