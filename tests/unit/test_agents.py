"""
Unit tests for SubAgentResult and MemorizeSubAgent.

V1.1: MemorizeSubAgent uses MemoryProvider instead of MemoryManager.
Post-write verification (content hash check) has been retired.
"""

from unittest.mock import MagicMock

import pytest

from src.agents.base import SubAgentResult
from src.agents.memorize import MemorizeSubAgent
from src.memory.stub_provider import StubMemoryProvider


# ---------------------------------------------------------------------------
# SubAgentResult
# ---------------------------------------------------------------------------


def test_subagent_result_success() -> None:
    """A successful SubAgentResult must expose success=True, the output, and error=None."""
    r: SubAgentResult = SubAgentResult(success=True, output="done")
    assert r.success is True
    assert r.output == "done"
    assert r.error is None


def test_subagent_result_failure() -> None:
    """A failed SubAgentResult must expose success=False and a non-None error string."""
    r: SubAgentResult = SubAgentResult(success=False, output="", error="oops")
    assert r.success is False
    assert r.error == "oops"


# ---------------------------------------------------------------------------
# MemorizeSubAgent helpers
# ---------------------------------------------------------------------------


def _make_agent() -> MemorizeSubAgent:
    """Build a MemorizeSubAgent with a StubMemoryProvider."""
    return MemorizeSubAgent(memory_provider=StubMemoryProvider())


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_valid_semantic_returns_success() -> None:
    """A valid semantic save must succeed and call save exactly once."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="I prefer dark mode.", memory_type="semantic", topic="prefs")
    assert result.success is True
    assert "Saved" in result.output
    assert len(agent.memory_provider._items) == 1


def test_valid_episodic_no_topic() -> None:
    """A valid episodic save without a topic must succeed."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="We met today.", memory_type="episodic")
    assert result.success is True
    items = list(agent.memory_provider._items.values())
    assert items[0].topic is None


def test_valid_procedural() -> None:
    """A valid procedural save must succeed regardless of topic."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="Run make deploy.", memory_type="procedural", topic="ops")
    assert result.success is True


def test_content_stripped_before_save() -> None:
    """Leading and trailing whitespace in content must be stripped before save."""
    agent: MemorizeSubAgent = _make_agent()
    agent.run(content="  hello  ", memory_type="semantic")
    items = list(agent.memory_provider._items.values())
    assert items[0].content == "hello"


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_empty_content_returns_failure() -> None:
    """An empty content string must return success=False without saving."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="", memory_type="semantic")
    assert result.success is False
    assert result.error is not None
    assert len(agent.memory_provider._items) == 0


def test_whitespace_content_returns_failure() -> None:
    """Whitespace-only content must be rejected with success=False."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="   ", memory_type="semantic")
    assert result.success is False
    assert len(agent.memory_provider._items) == 0


def test_invalid_type_returns_failure() -> None:
    """An unrecognised memory_type must return success=False with an informative error."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="hello", memory_type="invalid")
    assert result.success is False
    assert "invalid" in (result.error or "").lower()
    assert len(agent.memory_provider._items) == 0


def test_wrong_case_type_rejected() -> None:
    """MemorizeSubAgent does not normalise case — 'Semantic' must be rejected."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="hello", memory_type="Semantic")
    assert result.success is False


# ---------------------------------------------------------------------------
# Save failure
# ---------------------------------------------------------------------------


def test_save_failure_returns_failure() -> None:
    """A save() exception must be caught and surfaced as success=False."""
    agent: MemorizeSubAgent = _make_agent()
    agent.memory_provider.save = MagicMock(side_effect=OSError("disk full"))
    result: SubAgentResult = agent.run(content="hello", memory_type="semantic")
    assert result.success is False
    assert "disk full" in (result.error or "")
