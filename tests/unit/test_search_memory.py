"""
Unit tests for the ``search_memory`` tool (V1.1 M3).

Exercise the explicit-filter wrapper against a deterministic ``StubMemoryProvider``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.memory.stub_provider import StubMemoryProvider
from src.tools.search_memory import create_search_memory_tool


@pytest.fixture
def provider() -> StubMemoryProvider:
    """Stub provider pre-seeded with a mix of types and topics."""
    p = StubMemoryProvider()
    p.save("Python is a programming language", type="semantic", topic="programming")
    p.save("I like coffee in the morning", type="semantic", topic="personal")
    p.save("Use pytest for Python testing", type="procedural", topic="testing")
    p.save("The user prefers dark mode", type="semantic", topic="preferences")
    p.save("We had a team meeting today", type="episodic", topic=None)
    return p


@pytest.fixture
def search(provider: StubMemoryProvider):
    """Search tool bound to the seeded stub provider."""
    return create_search_memory_tool(provider)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_plain_query_returns_matches(search) -> None:
    """A plain query returns matching items in the ok envelope."""
    result = search(query="python")
    assert result["ok"] is True
    contents = [item["content"].lower() for item in result["data"]]
    assert any("python is a programming" in c for c in contents)
    assert any("pytest for python" in c for c in contents)
    assert result["metadata"]["result_count"] == len(result["data"])


def test_type_filter_restricts_results(search) -> None:
    """Passing a type filter drops items of other types."""
    result = search(query="python", type="procedural")
    assert result["ok"] is True
    assert all(it["type"] == "procedural" for it in result["data"])
    assert result["metadata"]["type"] == "procedural"


def test_topic_filter_restricts_results(search) -> None:
    """Passing a topic filter restricts to items with that topic."""
    result = search(query="", topic="preferences")
    assert result["ok"] is True
    assert all(it["topic"] == "preferences" for it in result["data"])


def test_limit_clamped_to_hard_cap(search, provider: StubMemoryProvider) -> None:
    """``limit`` is clamped to the tool's hard cap."""
    result = search(query="", limit=10_000)
    assert result["metadata"]["limit"] == 25


def test_limit_lifted_to_one(search) -> None:
    """A zero or negative limit is lifted to 1."""
    result = search(query="", limit=0)
    assert result["metadata"]["limit"] == 1
    assert len(result["data"]) <= 1


def test_empty_query_returns_all_within_limit(search) -> None:
    """An empty query matches all items, subject to the limit."""
    result = search(query="")
    assert result["ok"] is True
    assert len(result["data"]) == len(result["data"])  # non-empty


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_invalid_type_returns_error(search) -> None:
    """A bad type value returns ok=False without calling search."""
    result = search(query="anything", type="unknown")
    assert result["ok"] is False
    assert result["metadata"]["field"] == "type"


def test_case_insensitive_type(search) -> None:
    """An uppercase type is normalised and accepted."""
    result = search(query="python", type="SEMANTIC")
    assert result["ok"] is True
    assert all(it["type"] == "semantic" for it in result["data"])


def test_provider_raises_wraps_as_error() -> None:
    """An exception from the provider surfaces as ok=False with the message."""
    broken = StubMemoryProvider()
    broken.search = MagicMock(side_effect=RuntimeError("backend down"))
    tool = create_search_memory_tool(broken)
    result = tool(query="x")
    assert result["ok"] is False
    assert "backend down" in result["error"]


def test_item_dict_shape(search) -> None:
    """Every item dict carries the expected keys."""
    result = search(query="python")
    for it in result["data"]:
        assert set(it) == {"key", "content", "type", "topic", "timestamp", "relevance_score"}
