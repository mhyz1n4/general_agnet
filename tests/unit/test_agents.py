"""
Unit tests for SubAgentResult and MemorizeSubAgent.

Covers the happy path (all three memory types), validation failures (empty
content, wrong type, wrong case), save errors, and the verification step
(hash-found/not-found/exception paths).
MemoryManager is fully mocked so no filesystem I/O is required.
"""

import hashlib
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import SubAgentResult
from src.agents.memorize import MemorizeSubAgent


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


def _make_agent(find_return: Optional[str] = "abc123") -> MemorizeSubAgent:
    """
    Build a MemorizeSubAgent with a mocked MemoryManager.

    Args:
        find_return: Value that indexer.find_by_content_hash returns.
                     Use None to simulate a missing verification hash.
    """
    mm: MagicMock = MagicMock()
    mm.indexer.find_by_content_hash.return_value = find_return
    return MemorizeSubAgent(memory_manager=mm)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_valid_semantic_returns_success() -> None:
    """A valid semantic save must succeed and call save_message exactly once."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="I prefer dark mode.", memory_type="semantic", topic="prefs")
    assert result.success is True
    assert "verified" in result.output.lower()
    agent.memory_manager.save_message.assert_called_once()


def test_valid_episodic_no_topic() -> None:
    """A valid episodic save without a topic must succeed with no 'topic' key in metadata."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="We met today.", memory_type="episodic")
    assert result.success is True
    call_kwargs = agent.memory_manager.save_message.call_args
    metadata: dict = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else call_kwargs[1].get("metadata", {})
    assert "topic" not in metadata


def test_valid_procedural() -> None:
    """A valid procedural save must succeed regardless of whether a topic is provided."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="Run make deploy.", memory_type="procedural", topic="ops")
    assert result.success is True


def test_content_stripped_before_save() -> None:
    """Leading and trailing whitespace in content must be stripped before save_message is called."""
    agent: MemorizeSubAgent = _make_agent()
    agent.run(content="  hello  ", memory_type="semantic")
    saved_content: str = agent.memory_manager.save_message.call_args[0][1]
    assert saved_content == "hello"


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_empty_content_returns_failure() -> None:
    """An empty content string must return success=False without calling save_message."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="", memory_type="semantic")
    assert result.success is False
    assert result.error is not None
    agent.memory_manager.save_message.assert_not_called()


def test_whitespace_content_returns_failure() -> None:
    """Whitespace-only content must be rejected with success=False."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="   ", memory_type="semantic")
    assert result.success is False
    agent.memory_manager.save_message.assert_not_called()


def test_invalid_type_returns_failure() -> None:
    """An unrecognised memory_type must return success=False with an informative error."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="hello", memory_type="invalid")
    assert result.success is False
    assert "invalid" in (result.error or "").lower()
    agent.memory_manager.save_message.assert_not_called()


def test_wrong_case_type_rejected() -> None:
    """MemorizeSubAgent does not normalise case — 'Semantic' must be rejected."""
    agent: MemorizeSubAgent = _make_agent()
    result: SubAgentResult = agent.run(content="hello", memory_type="Semantic")
    assert result.success is False


# ---------------------------------------------------------------------------
# Save failure
# ---------------------------------------------------------------------------


def test_save_failure_returns_failure() -> None:
    """A save_message() exception must be caught and surfaced as success=False with the error message."""
    agent: MemorizeSubAgent = _make_agent()
    agent.memory_manager.save_message.side_effect = OSError("disk full")
    result: SubAgentResult = agent.run(content="hello", memory_type="semantic")
    assert result.success is False
    assert "disk full" in (result.error or "")


# ---------------------------------------------------------------------------
# Verification inconclusive path
# ---------------------------------------------------------------------------


def test_verification_inconclusive_still_success() -> None:
    """If the hash is not found in the index after save, result is still success=True with a note."""
    agent: MemorizeSubAgent = _make_agent(find_return=None)
    result: SubAgentResult = agent.run(content="hello", memory_type="semantic")
    assert result.success is True
    assert "inconclusive" in result.output.lower()


def test_verification_exception_still_success() -> None:
    """An exception during verification must be swallowed; save was successful so return success=True."""
    agent: MemorizeSubAgent = _make_agent()
    agent.memory_manager.indexer.find_by_content_hash.side_effect = RuntimeError("db gone")
    result: SubAgentResult = agent.run(content="hello", memory_type="semantic")
    assert result.success is True


# ---------------------------------------------------------------------------
# Correct content hash is passed to find_by_content_hash
# ---------------------------------------------------------------------------


def test_correct_hash_verified() -> None:
    """The SHA-256[:16] hash of the content must be passed to find_by_content_hash for verification."""
    agent: MemorizeSubAgent = _make_agent()
    content: str = "check hash"
    agent.run(content=content, memory_type="semantic")
    expected_hash: str = hashlib.sha256(content.encode()).hexdigest()[:16]
    agent.memory_manager.indexer.find_by_content_hash.assert_called_once_with(expected_hash)
