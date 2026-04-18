"""
Unit tests for the memorize tool (src/tools/memorize.py).

V1.1: Uses StubMemoryProvider instead of mocked MemoryManager.
Covers valid inputs, invalid inputs, case normalisation, save failure
recovery, and content integrity.
"""

from unittest.mock import MagicMock

import pytest

from src.tools.memorize import create_memorize_tool
from src.memory.stub_provider import StubMemoryProvider
from src.constants import VALID_MEMORY_TYPES as VALID_TYPES


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
    """A valid episodic save must return a confirmation with the key."""
    result: str = memorize(content="Had a meeting", type="episodic")
    assert "Saved to memory" in result
    assert len(provider._items) == 1


def test_valid_semantic_with_topic(memorize, provider: StubMemoryProvider) -> None:
    """A semantic save with a topic must store the topic in the item."""
    memorize(content="I prefer Python", type="semantic", topic="preferences")
    items = list(provider._items.values())
    assert len(items) == 1
    assert items[0].topic == "preferences"
    assert items[0].type == "semantic"


def test_valid_procedural(memorize, provider: StubMemoryProvider) -> None:
    """A valid procedural save must succeed and return a confirmation string."""
    result: str = memorize(content="Step 1: ...", type="procedural")
    assert "Saved" in result


def test_empty_topic_stored_as_none(memorize, provider: StubMemoryProvider) -> None:
    """An explicitly empty topic string must be stored as None."""
    memorize(content="some fact", type="semantic", topic="")
    items = list(provider._items.values())
    assert items[0].topic is None


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_empty_content_returns_error(memorize, provider: StubMemoryProvider) -> None:
    """An empty content string must return an error without saving."""
    result: str = memorize(content="", type="episodic")
    assert "Error" in result
    assert len(provider._items) == 0


def test_whitespace_only_content_returns_error(memorize, provider: StubMemoryProvider) -> None:
    """Whitespace-only content must be rejected without saving."""
    result: str = memorize(content="   \n\t  ", type="episodic")
    assert "Error" in result
    assert len(provider._items) == 0


def test_invalid_type_returns_error(memorize, provider: StubMemoryProvider) -> None:
    """An unrecognised memory type must return an error listing valid types."""
    result: str = memorize(content="data", type="unknown_type")
    assert "Error" in result
    assert len(provider._items) == 0


def test_wrong_case_type_normalised(memorize, provider: StubMemoryProvider) -> None:
    """casefold normalises 'Episodic' -> 'episodic' silently; save must succeed."""
    result: str = memorize(content="data", type="Episodic")
    assert "Error" not in result
    assert len(provider._items) == 1


# ---------------------------------------------------------------------------
# Error recovery
# ---------------------------------------------------------------------------


def test_save_raises_returns_error_string(provider: StubMemoryProvider) -> None:
    """A save() exception must be caught; the tool must return an error string."""
    provider.save = MagicMock(side_effect=OSError("disk full"))
    memorize = create_memorize_tool(provider)
    result: str = memorize(content="hello", type="semantic")
    assert "Error" in result
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Content integrity
# ---------------------------------------------------------------------------


def test_unicode_content_passed_through(memorize, provider: StubMemoryProvider) -> None:
    """Unicode content must survive the save step intact."""
    content: str = "emoji 🎉 and Chinese 中文"
    memorize(content=content, type="semantic")
    items = list(provider._items.values())
    assert content in items[0].content


def test_content_is_stripped(memorize, provider: StubMemoryProvider) -> None:
    """Leading/trailing whitespace must be stripped from content before saving."""
    memorize(content="  hello  ", type="semantic")
    items = list(provider._items.values())
    assert items[0].content == "hello"
