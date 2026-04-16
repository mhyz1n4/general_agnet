"""
Unit tests for the memorize tool (src/tools/memorize.py).

Covers valid inputs (all three memory types, topic handling), invalid inputs
(empty content, bad type), case normalisation via casefold(), save failure
recovery, and content integrity after the timestamp-prefix enrichment.
MemoryManager is fully mocked so no filesystem I/O is needed.
"""

from unittest.mock import MagicMock

import pytest

from src.tools.memorize import create_memorize_tool
from src.constants import VALID_MEMORY_TYPES as VALID_TYPES


@pytest.fixture
def mock_mm() -> MagicMock:
    """Mock MemoryManager whose save_message succeeds by default."""
    return MagicMock()


@pytest.fixture
def memorize(mock_mm: MagicMock):
    """Memorize tool function bound to the mock MemoryManager."""
    return create_memorize_tool(mock_mm)


# ---------------------------------------------------------------------------
# Valid inputs
# ---------------------------------------------------------------------------


def test_valid_episodic_saves_and_returns_key(memorize, mock_mm: MagicMock) -> None:
    """A valid episodic save must return a key containing 'episodic_' and confirm success."""
    result: str = memorize(content="Had a meeting", type="episodic")
    assert "episodic_" in result
    assert "Saved to memory" in result
    mock_mm.save_message.assert_called_once()


def test_valid_semantic_with_topic(memorize, mock_mm: MagicMock) -> None:
    """A semantic save with a topic must store the topic and type in the metadata."""
    memorize(content="I prefer Python", type="semantic", topic="preferences")
    _, _, meta = mock_mm.save_message.call_args[0]
    assert meta["topic"] == "preferences"
    assert meta["type"] == "semantic"


def test_valid_procedural(memorize, mock_mm: MagicMock) -> None:
    """A valid procedural save must succeed and return a confirmation string."""
    result: str = memorize(content="Step 1: ...", type="procedural")
    assert "Saved" in result


def test_empty_topic_stored_as_none(memorize, mock_mm: MagicMock) -> None:
    """An explicitly empty topic string must be stored as None in the metadata."""
    memorize(content="some fact", type="semantic", topic="")
    _, _, meta = mock_mm.save_message.call_args[0]
    assert meta["topic"] is None


def test_whitespace_topic_stored_as_none(memorize, mock_mm: MagicMock) -> None:
    """A whitespace-only topic string must be normalised to None in the metadata."""
    memorize(content="some fact", type="semantic", topic="   ")
    _, _, meta = mock_mm.save_message.call_args[0]
    assert meta["topic"] is None


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_empty_content_returns_error(memorize, mock_mm: MagicMock) -> None:
    """An empty content string must return an error message without calling save_message."""
    result: str = memorize(content="", type="episodic")
    assert "Error" in result
    mock_mm.save_message.assert_not_called()


def test_whitespace_only_content_returns_error(memorize, mock_mm: MagicMock) -> None:
    """Whitespace-only content must be rejected without calling save_message."""
    result: str = memorize(content="   \n\t  ", type="episodic")
    assert "Error" in result
    mock_mm.save_message.assert_not_called()


def test_invalid_type_returns_error(memorize, mock_mm: MagicMock) -> None:
    """An unrecognised memory type must return an error listing valid types."""
    result: str = memorize(content="data", type="unknown_type")
    assert "Error" in result
    assert "episodic" in result or "semantic" in result
    mock_mm.save_message.assert_not_called()


def test_wrong_case_type_normalised(memorize, mock_mm: MagicMock) -> None:
    """casefold normalises 'Episodic' → 'episodic' silently; save must succeed."""
    result: str = memorize(content="data", type="Episodic")
    assert "Error" not in result
    mock_mm.save_message.assert_called_once()


# ---------------------------------------------------------------------------
# Error recovery
# ---------------------------------------------------------------------------


def test_save_message_raises_returns_error_string(memorize, mock_mm: MagicMock) -> None:
    """A save_message() exception must be caught; the tool must return an error string, not raise."""
    mock_mm.save_message.side_effect = OSError("disk full")
    result: str = memorize(content="hello", type="semantic")
    assert "Error" in result
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Content integrity
# ---------------------------------------------------------------------------


def test_unicode_content_passed_through(memorize, mock_mm: MagicMock) -> None:
    """Unicode content must survive the timestamp-enrichment step intact."""
    content: str = "emoji 🎉 and Chinese 中文"
    memorize(content=content, type="semantic")
    saved_content: str = mock_mm.save_message.call_args[0][1]
    assert content in saved_content


def test_very_long_content_not_truncated(memorize, mock_mm: MagicMock) -> None:
    """A 10 000-character content string must not be truncated during save."""
    content: str = "x" * 10_000
    memorize(content=content, type="episodic")
    saved_content: str = mock_mm.save_message.call_args[0][1]
    assert content in saved_content


def test_content_is_stripped(memorize, mock_mm: MagicMock) -> None:
    """Leading/trailing whitespace must be stripped from content before saving."""
    memorize(content="  hello  ", type="semantic")
    saved_content: str = mock_mm.save_message.call_args[0][1]
    assert "hello" in saved_content
    assert saved_content == saved_content.strip()
