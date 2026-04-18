"""
Unit tests for ReMeCompactionManager.

Verifies compaction lifecycle, tool-pair safety, empty-summary fallback,
and aggressive reduction on context overflow.
"""

from unittest.mock import MagicMock

import pytest

from src.memory.provider import Summary
from src.memory.remelight.compaction_manager import (
    ReMeCompactionManager,
    _has_tool_result,
    _has_tool_use,
    _safe_split_index,
    _strands_to_provider_messages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: str, text: str) -> dict:
    """Build a minimal Strands-format message."""
    return {"role": role, "content": [{"text": text}]}


def _tool_use_msg() -> dict:
    """Build a message containing a toolUse block."""
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "memorize", "input": {}}}],
    }


def _tool_result_msg() -> dict:
    """Build a message containing a toolResult block."""
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "ok"}]}}],
    }


def _make_manager(
    window_size: int = 5,
    compact_batch_size: int = 3,
    preserve_recent: int = 2,
    summary_text: str = "summary of old messages",
) -> tuple[ReMeCompactionManager, MagicMock]:
    """Return a manager and its mock provider."""
    provider = MagicMock()
    provider.compact.return_value = Summary(text=summary_text, source_count=0)
    provider.save.return_value = MagicMock()
    mgr = ReMeCompactionManager(
        memory_provider=provider,
        window_size=window_size,
        compact_batch_size=compact_batch_size,
        preserve_recent=preserve_recent,
    )
    return mgr, provider


def _make_agent(messages: list[dict]) -> MagicMock:
    """Return a mock Agent with the given messages list."""
    agent = MagicMock()
    agent.messages = messages
    return agent


# ---------------------------------------------------------------------------
# Content-block helpers
# ---------------------------------------------------------------------------

class TestContentBlockHelpers:
    """Verify _has_tool_use and _has_tool_result detection."""

    def test_has_tool_use_true(self):
        assert _has_tool_use(_tool_use_msg()) is True

    def test_has_tool_use_false(self):
        assert _has_tool_use(_msg("user", "hello")) is False

    def test_has_tool_result_true(self):
        assert _has_tool_result(_tool_result_msg()) is True

    def test_has_tool_result_false(self):
        assert _has_tool_result(_msg("assistant", "hi")) is False


# ---------------------------------------------------------------------------
# _safe_split_index
# ---------------------------------------------------------------------------

class TestSafeSplitIndex:
    """Verify split-point adjustment for tool-pair safety."""

    def test_no_tools_returns_raw(self):
        msgs = [_msg("user", "a"), _msg("assistant", "b"), _msg("user", "c")]
        assert _safe_split_index(msgs, 1) == 1

    def test_skips_tool_result_at_boundary(self):
        msgs = [_msg("user", "a"), _tool_result_msg(), _msg("user", "c")]
        assert _safe_split_index(msgs, 1) == 2

    def test_skips_orphaned_tool_use(self):
        msgs = [_msg("user", "a"), _tool_use_msg(), _msg("user", "c")]
        assert _safe_split_index(msgs, 1) == 2

    def test_keeps_tool_use_with_following_result(self):
        msgs = [_msg("user", "a"), _tool_use_msg(), _tool_result_msg(), _msg("user", "d")]
        assert _safe_split_index(msgs, 1) == 1

    def test_zero_split_returns_zero(self):
        msgs = [_msg("user", "a")]
        assert _safe_split_index(msgs, 0) == 0


# ---------------------------------------------------------------------------
# _strands_to_provider_messages
# ---------------------------------------------------------------------------

class TestStrandsToProviderMessages:
    """Verify Strands → MemoryProvider message conversion."""

    def test_text_extraction(self):
        msgs = [_msg("user", "hello"), _msg("assistant", "hi")]
        result = _strands_to_provider_messages(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    def test_multi_block_joined(self):
        msg = {"role": "assistant", "content": [{"text": "a"}, {"text": "b"}]}
        result = _strands_to_provider_messages([msg])
        assert result[0]["content"] == "a\nb"

    def test_non_text_blocks_skipped(self):
        msg = {"role": "assistant", "content": [{"toolUse": {"name": "x"}}, {"text": "visible"}]}
        result = _strands_to_provider_messages([msg])
        assert result[0]["content"] == "visible"


# ---------------------------------------------------------------------------
# apply_management — no-op under threshold
# ---------------------------------------------------------------------------

class TestApplyManagementNoOp:
    """apply_management should be a no-op when under window_size."""

    def test_under_threshold_no_compact(self):
        mgr, provider = _make_manager(window_size=10)
        agent = _make_agent([_msg("user", f"m{i}") for i in range(5)])
        mgr.apply_management(agent)
        provider.compact.assert_not_called()
        assert len(agent.messages) == 5

    def test_at_threshold_no_compact(self):
        mgr, provider = _make_manager(window_size=5)
        agent = _make_agent([_msg("user", f"m{i}") for i in range(5)])
        mgr.apply_management(agent)
        provider.compact.assert_not_called()


# ---------------------------------------------------------------------------
# apply_management — compaction
# ---------------------------------------------------------------------------

class TestApplyManagementCompaction:
    """apply_management should compact and persist when over window_size."""

    def test_compact_replaces_with_summary(self):
        mgr, provider = _make_manager(window_size=5, compact_batch_size=3, preserve_recent=2)
        messages = [_msg("user", f"m{i}") for i in range(8)]
        agent = _make_agent(messages)

        mgr.apply_management(agent)

        provider.compact.assert_called_once()
        provider.save.assert_called_once()
        save_call = provider.save.call_args
        assert save_call.kwargs.get("type") == "episodic" or save_call.args[1] == "episodic"

        assert agent.messages[0]["content"][0]["text"].startswith("[Previous conversation summary]")
        assert len(agent.messages) <= 8

    def test_removed_message_count_tracked(self):
        mgr, provider = _make_manager(window_size=5, compact_batch_size=3, preserve_recent=2)
        agent = _make_agent([_msg("user", f"m{i}") for i in range(8)])
        mgr.apply_management(agent)
        assert mgr.removed_message_count > 0


# ---------------------------------------------------------------------------
# apply_management — empty summary fallback
# ---------------------------------------------------------------------------

class TestEmptySummaryFallback:
    """When compact returns empty text, fall back to sliding-window drop."""

    def test_empty_summary_drops_messages(self):
        mgr, provider = _make_manager(
            window_size=5, compact_batch_size=3, preserve_recent=2, summary_text=""
        )
        messages = [_msg("user", f"m{i}") for i in range(8)]
        agent = _make_agent(messages)

        mgr.apply_management(agent)

        provider.save.assert_not_called()
        assert len(agent.messages) < 8
        assert mgr.removed_message_count > 0

    def test_compact_exception_falls_back_to_drop(self):
        mgr, provider = _make_manager(window_size=5, compact_batch_size=3)
        provider.compact.side_effect = RuntimeError("LLM down")
        messages = [_msg("user", f"m{i}") for i in range(8)]
        agent = _make_agent(messages)

        mgr.apply_management(agent)

        assert len(agent.messages) < 8
        assert mgr.removed_message_count > 0


# ---------------------------------------------------------------------------
# apply_management — tool-pair safety
# ---------------------------------------------------------------------------

class TestToolPairSafety:
    """Compaction must not orphan toolUse/toolResult pairs."""

    def test_tool_pair_at_boundary_kept_together(self):
        mgr, provider = _make_manager(window_size=4, compact_batch_size=2, preserve_recent=2)
        messages = [
            _msg("user", "m0"),
            _msg("user", "m1"),
            _tool_use_msg(),
            _tool_result_msg(),
            _msg("user", "m4"),
            _msg("user", "m5"),
        ]
        agent = _make_agent(messages)
        mgr.apply_management(agent)

        for msg in agent.messages:
            if _has_tool_result(msg):
                idx = agent.messages.index(msg)
                assert idx > 0, "toolResult should not be the first message"


# ---------------------------------------------------------------------------
# flush
# ---------------------------------------------------------------------------


class TestFlush:
    """flush should compact any remaining messages at session end."""

    def test_flush_compacts_all_messages(self):
        """Non-empty message list is fully compacted into a single summary message."""
        mgr, provider = _make_manager(window_size=100)
        messages = [_msg("user", f"m{i}") for i in range(6)]
        agent = _make_agent(messages)

        mgr.flush(agent)

        provider.compact.assert_called_once()
        provider.save.assert_called_once()
        assert len(agent.messages) == 1
        assert agent.messages[0]["content"][0]["text"].startswith(
            "[Previous conversation summary]"
        )

    def test_flush_empty_messages_is_noop(self):
        """Empty message list must not invoke the provider."""
        mgr, provider = _make_manager()
        agent = _make_agent([])

        mgr.flush(agent)

        provider.compact.assert_not_called()
        provider.save.assert_not_called()
        assert agent.messages == []

    def test_flush_empty_summary_drops_all(self):
        """Stub LLM (empty summary) must fall back to dropping the messages."""
        mgr, provider = _make_manager(summary_text="")
        agent = _make_agent([_msg("user", f"m{i}") for i in range(4)])

        mgr.flush(agent)

        provider.save.assert_not_called()
        assert agent.messages == []


# ---------------------------------------------------------------------------
# reduce_context
# ---------------------------------------------------------------------------

class TestReduceContext:
    """reduce_context should aggressively compact on overflow."""

    def test_aggressive_reduction(self):
        mgr, provider = _make_manager(window_size=5, preserve_recent=2)
        messages = [_msg("user", f"m{i}") for i in range(10)]
        agent = _make_agent(messages)

        mgr.reduce_context(agent)

        assert len(agent.messages) <= 3  # summary + 2 preserved
        provider.compact.assert_called_once()

    def test_already_minimal_raises(self):
        from strands.types.exceptions import ContextWindowOverflowException

        mgr, _ = _make_manager(preserve_recent=5)
        agent = _make_agent([_msg("user", "m0")])

        with pytest.raises(ContextWindowOverflowException):
            mgr.reduce_context(agent)
