"""
Shared fixtures and test data for integration tests.

V1.1: Memory stack uses ``StubMemoryProvider`` for fast isolated tests.
``ReMeLightProvider`` is available for integration tests that need it.

Fixtures
--------
mem_root     — isolated tmp directory for each test's memory files
memory_provider — StubMemoryProvider instance
session_id   — unique session ID per test
strands_patch — patches Strands SDK to handle vLLM streaming quirks (#815)
agent_or_mock — real Strands Agent when LLM is reachable, else a MagicMock

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
# Strands SDK patch fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def strands_patch():
    """
    Patch Strands SDK to handle vLLM streaming quirks (strands issue #815).

    Restores the original function after each test.
    """
    try:
        import strands.tools.tools as _strands_tools
        import strands.event_loop.streaming as _strands_streaming
        from strands.tools.tools import InvalidToolUseNameException
        from src.orchestrator import tool_call_counter

        _orig = _strands_tools.validate_tool_use_name

        def _safe(tool: dict) -> None:
            """Guarded replacement for ``validate_tool_use_name``."""
            count = getattr(tool_call_counter, "count", 0) + 1
            limit = getattr(tool_call_counter, "limit", 0)
            tool_call_counter.count = count
            if limit and count > limit:
                raise InvalidToolUseNameException(
                    f"tool call limit reached ({count}/{limit})"
                )
            if not tool.get("name"):
                raise InvalidToolUseNameException(
                    "tool name is None or empty (strands #815)"
                )
            _orig(tool)

        _strands_tools.validate_tool_use_name = _safe
        _strands_streaming.validate_tool_use_name = _safe
        yield
        _strands_tools.validate_tool_use_name = _orig
        _strands_streaming.validate_tool_use_name = _orig
    except ImportError:
        yield


# ---------------------------------------------------------------------------
# Agent fixture — real or mock depending on LLM availability
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_or_mock(llm_endpoint_or_none, strands_patch):
    """
    Return a real Strands Agent when the LLM endpoint is reachable, else a
    ``MagicMock`` that returns a fixed response string.
    """
    if llm_endpoint_or_none is None:
        return MagicMock(return_value="This is the assistant response.")

    try:
        from strands import Agent
        from strands.models.openai import OpenAIModel
    except ImportError:
        return MagicMock(return_value="This is the assistant response.")

    from src.config import Config
    from src.orchestrator import tool_call_counter

    config = Config()
    tool_call_counter.count = 0
    tool_call_counter.limit = config.max_tool_calls

    model = OpenAIModel(
        client_args={
            "api_key": config.llm_api_key,
            "base_url": config.llm_api_endpoint,
        },
        model_id=config.llm_model,
        params={"max_tokens": max(config.llm_max_tokens, 4096)},
    )
    return Agent(model=model, tools=[], callback_handler=None)
