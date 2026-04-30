"""
Unit tests for the memorize tool (src/tools/memorize.py).

V1.1 M1: tools now return the ``ToolResult`` envelope
(``{ok, data, error, metadata}``) rather than plain strings.  Tests assert on
the envelope fields so regressions in the shape itself surface immediately.
"""

from unittest.mock import MagicMock

import pytest

from src.constants import VALID_MEMORY_TYPES as VALID_TYPES
from src.memory.stub_provider import StubMemoryProvider
from src.tools.memorize import create_memorize_tool


@pytest.fixture
def provider() -> StubMemoryProvider:
    """Fresh StubMemoryProvider for each test."""
    return StubMemoryProvider()


@pytest.fixture
def memorize(provider: StubMemoryProvider):
    """Memorize tool function bound to the StubMemoryProvider."""
    return create_memorize_tool(provider)


# ---------------------------------------------------------------------------
# Valid inputs
# ---------------------------------------------------------------------------


def test_valid_episodic_saves_and_returns_key(memorize, provider: StubMemoryProvider) -> None:
    """A valid episodic save returns ok=True with the saved key in data."""
    result = memorize(content="Had a meeting", type="episodic")
    assert result["ok"] is True
    assert result["error"] is None
    assert result["data"]["key"]
    assert len(provider._items) == 1


def test_valid_semantic_with_topic(memorize, provider: StubMemoryProvider) -> None:
    """A semantic save with a topic stores the topic on the item and echoes it in metadata."""
    result = memorize(content="I prefer Python", type="semantic", topic="preferences")
    assert result["ok"] is True
    assert result["metadata"]["topic"] == "preferences"
    items = list(provider._items.values())
    assert items[0].topic == "preferences"
    assert items[0].type == "semantic"


def test_valid_procedural(memorize, provider: StubMemoryProvider) -> None:
    """A valid procedural save returns ok=True."""
    result = memorize(content="Step 1: ...", type="procedural")
    assert result["ok"] is True
    assert result["metadata"]["type"] == "procedural"


def test_empty_topic_stored_as_none(memorize, provider: StubMemoryProvider) -> None:
    """An explicitly empty topic string is stored as None on the item."""
    memorize(content="some fact", type="semantic", topic="")
    items = list(provider._items.values())
    assert items[0].topic is None


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_empty_content_returns_error(memorize, provider: StubMemoryProvider) -> None:
    """Empty content returns ok=False with an error message and no save."""
    result = memorize(content="", type="episodic")
    assert result["ok"] is False
    assert "empty" in result["error"]
    assert len(provider._items) == 0


def test_whitespace_only_content_returns_error(memorize, provider: StubMemoryProvider) -> None:
    """Whitespace-only content is rejected without saving."""
    result = memorize(content="   \n\t  ", type="episodic")
    assert result["ok"] is False
    assert result["error"]
    assert len(provider._items) == 0


def test_invalid_type_returns_error(memorize, provider: StubMemoryProvider) -> None:
    """An unrecognised memory type returns ok=False listing valid types."""
    result = memorize(content="data", type="unknown_type")
    assert result["ok"] is False
    assert "type must be one of" in result["error"]
    assert len(provider._items) == 0


def test_wrong_case_type_normalised(memorize, provider: StubMemoryProvider) -> None:
    """casefold normalises 'Episodic' -> 'episodic' silently; save succeeds."""
    result = memorize(content="data", type="Episodic")
    assert result["ok"] is True
    assert len(provider._items) == 1


# ---------------------------------------------------------------------------
# Error recovery
# ---------------------------------------------------------------------------


def test_save_raises_returns_error_envelope(provider: StubMemoryProvider) -> None:
    """A save() exception surfaces as ok=False with the provider error in the message."""
    provider.save = MagicMock(side_effect=OSError("disk full"))
    memorize = create_memorize_tool(provider)
    result = memorize(content="hello", type="semantic")
    assert result["ok"] is False
    assert "disk full" in result["error"]


# ---------------------------------------------------------------------------
# Content integrity
# ---------------------------------------------------------------------------


def test_unicode_content_passed_through(memorize, provider: StubMemoryProvider) -> None:
    """Unicode content survives the save step intact."""
    content: str = "emoji 🎉 and Chinese 中文"
    memorize(content=content, type="semantic")
    items = list(provider._items.values())
    assert content in items[0].content


def test_content_is_stripped(memorize, provider: StubMemoryProvider) -> None:
    """Leading/trailing whitespace is stripped from content before saving."""
    memorize(content="  hello  ", type="semantic")
    items = list(provider._items.values())
    assert items[0].content == "hello"
