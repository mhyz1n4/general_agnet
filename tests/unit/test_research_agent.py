"""
Unit tests for the research sub-agent factory (V1.1 M5).

The live sub-agent requires Strands + a reachable LLM, so here we only verify
the wiring: the factory returns a Strands tool-shaped object with the
expected name when construction succeeds, and returns None gracefully when
the Strands import fails.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.memory.stub_provider import StubMemoryProvider


@pytest.fixture
def cfg() -> SimpleNamespace:
    """Minimal config stand-in with the fields the factory touches."""
    return SimpleNamespace(
        llm_api_key="EMPTY",
        llm_api_endpoint="http://localhost:8000/v1",
        llm_model="/model",
        llm_max_tokens=2048,
        tavily_search_token=None,
        tavily_search_endpoint="https://api.tavily.com/search",
        tavily_search_timeout_seconds=10,
        sub_agent_max_tool_calls=3,
        sub_agent_timeout_seconds=30,
    )


def test_factory_returns_none_when_strands_missing(cfg) -> None:
    """If Strands cannot be imported, the factory logs and returns None."""
    from src.agents import research

    fake_strands = types.ModuleType("strands")
    # Force ImportError by leaving out the Agent attribute.
    with patch.dict(sys.modules, {"strands": fake_strands}, clear=False):
        result = research.create_research_agent_tool(StubMemoryProvider(), cfg)

    assert result is None


def test_factory_builds_tool_with_expected_name(cfg) -> None:
    """Happy path: factory returns a tool whose ``tool_name`` is delegate_to_research."""
    from src.agents import research

    tool = research.create_research_agent_tool(StubMemoryProvider(), cfg)
    assert tool is not None
    # The sub-agent is wrapped in a ``@tool``-decorated function so that the
    # parent agent sees it as a plain Strands tool with its own per-call
    # worker thread (see src/agents/research.py for the isolation rationale).
    assert tool.tool_name == research.DELEGATE_TOOL_NAME
    assert tool.tool_type == "function"
