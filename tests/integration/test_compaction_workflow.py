"""
Verify the conversation-compaction workflow with a simulated dialog.

Runs a ~16-turn session through ``ReMeCompactionManager`` backed by a tiny
fake-LLM provider that returns a short summary, then prints a before/after
size table and asserts the message count and character budget both shrink.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from src.constants import CHARS_PER_TOKEN
from src.memory.provider import MemoryItem, Message, SearchFilters, Summary
from src.memory.remelight.compaction_manager import ReMeCompactionManager


def _text(role: str, text: str) -> dict:
    """Build a Strands-format text message."""
    return {"role": role, "content": [{"text": text}]}


def _conversation() -> list[dict]:
    """Return a simulated debugging dialog."""
    return [
        _text("user", "My /checkout endpoint started returning 500s about an hour ago."),
        _text("assistant", "I'll check the logs for the checkout service."),
        _text("user", "Logs show 483 TypeError hits at handlers/checkout.py:142."),
        _text("assistant", "The refactor commit 4a1f9c2 landed just before the spike."),
        _text("user", "Show me the diff."),
        _text("assistant", "The new cart_service.get returns None for empty carts; old code assumed dict."),
        _text("user", "How do we confirm it's the empty-cart path?"),
        _text("assistant", "94% of error requests had zero cart items in the audit DB."),
        _text("user", "Draft the fix."),
        _text("assistant", "Guard with `items = cart.items if cart else []` and add an empty-cart unit test."),
        _text("user", "Save this incident so we don't re-diagnose next time."),
        _text("assistant", "Saved episodic memory under topic 'incidents'."),
        _text("user", "Also add a procedural note about the investigation recipe."),
        _text("assistant", "Saved: read_logs -> git_log -> correlate deploy with error onset."),
        _text("user", "Thanks, ping me once CI is green."),
        _text("assistant", "Will do."),
    ]


class _FakeLLMProvider:
    """Stand-in MemoryProvider whose compact returns a short fixed summary."""

    _SUMMARY = (
        "Debugged /checkout 500s: commit 4a1f9c2 refactor made cart_service.get "
        "return None for empty carts, unguarded subscript at handlers/checkout.py:142 "
        "caused the TypeError. Fix: items = cart.items if cart else []; saved incident "
        "and investigation recipe to memory."
    )

    def __init__(self) -> None:
        """Track items saved during compaction for later assertions."""
        self.saved: list[MemoryItem] = []

    def compact(self, messages: list[Message]) -> Summary:
        """Return a short fixed summary — stands in for a real LLM."""
        return Summary(text=self._SUMMARY, source_count=len(messages))

    def save(self, content: str, type: str, topic: Optional[str] = None) -> MemoryItem:
        """Record saves in-memory for assertion purposes."""
        item = MemoryItem(key=f"k{len(self.saved)}", content=content, type=type, topic=topic)  # type: ignore[arg-type]
        self.saved.append(item)
        return item

    def search(self, query: str, filters: SearchFilters) -> list[MemoryItem]:
        """Unused here."""
        raise NotImplementedError

    def check_context(self, messages: list[Message], budget_tokens: int) -> list[Message]:
        """Unused here."""
        raise NotImplementedError


def _measure(messages: list[dict]) -> tuple[int, int]:
    """Return (message_count, total_chars) across all text blocks."""
    chars = sum(
        len(b.get("text", ""))
        for m in messages
        for b in m.get("content", [])
        if "text" in b
    )
    return len(messages), chars


@pytest.mark.integration
def test_compaction_shrinks_conversation():
    """Compaction must shrink the live message list and preserve continuity."""
    provider = _FakeLLMProvider()
    manager = ReMeCompactionManager(
        memory_provider=provider,
        window_size=8,
        compact_batch_size=10,
        preserve_recent=4,
    )
    agent = MagicMock()
    agent.messages = _conversation()

    before_msgs, before_chars = _measure(agent.messages)
    manager.apply_management(agent)
    after_msgs, after_chars = _measure(agent.messages)

    print("\n=== Compaction report ===")
    print(f"messages: {before_msgs} -> {after_msgs}")
    print(f"chars:    {before_chars} -> {after_chars}  ({after_chars / before_chars:.0%} retained)")
    print(f"~tokens:  {before_chars // CHARS_PER_TOKEN} -> {after_chars // CHARS_PER_TOKEN}")
    print(f"summary:  {agent.messages[0]['content'][0]['text'][:180]}...")
    print("\n=== How compaction helps ===")
    print("- Keeps the live message list under the context window so the next turn fits.")
    print("- Cuts per-turn tokens (and thus latency + cost) by the same ratio.")
    print("- Preserves continuity via '[Previous conversation summary]' instead of dropping history.")
    print("- Persists the summary as episodic memory for retrieval in later sessions.")

    assert after_msgs < before_msgs
    assert after_chars < before_chars
    assert agent.messages[0]["content"][0]["text"].startswith("[Previous conversation summary]")
    assert len(provider.saved) == 1
    assert provider.saved[0].type == "episodic"
    assert provider.saved[0].topic == "compacted_dialog"
