"""Unit tests for the memorize tool."""

import pytest
from unittest.mock import MagicMock, patch

from src.tools.memorize import create_memorize_tool, VALID_TYPES


@pytest.fixture
def mock_mm():
    return MagicMock()


@pytest.fixture
def memorize(mock_mm):
    return create_memorize_tool(mock_mm)


# ---------------------------------------------------------------------------
# Valid inputs
# ---------------------------------------------------------------------------

def test_valid_episodic_saves_and_returns_key(memorize, mock_mm):
    result = memorize(content="Had a meeting", type="episodic")
    assert "episodic_" in result
    assert "Saved to memory" in result
    mock_mm.save_message.assert_called_once()


def test_valid_semantic_with_topic(memorize, mock_mm):
    memorize(content="I prefer Python", type="semantic", topic="preferences")
    _, _, meta = mock_mm.save_message.call_args[0]
    assert meta["topic"] == "preferences"
    assert meta["type"] == "semantic"


def test_valid_procedural(memorize, mock_mm):
    result = memorize(content="Step 1: ...", type="procedural")
    assert "Saved" in result


def test_empty_topic_stored_as_none(memorize, mock_mm):
    memorize(content="some fact", type="semantic", topic="")
    _, _, meta = mock_mm.save_message.call_args[0]
    assert meta["topic"] is None


def test_whitespace_topic_stored_as_none(memorize, mock_mm):
    memorize(content="some fact", type="semantic", topic="   ")
    _, _, meta = mock_mm.save_message.call_args[0]
    assert meta["topic"] is None


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_empty_content_returns_error(memorize, mock_mm):
    result = memorize(content="", type="episodic")
    assert "Error" in result
    mock_mm.save_message.assert_not_called()


def test_whitespace_only_content_returns_error(memorize, mock_mm):
    result = memorize(content="   \n\t  ", type="episodic")
    assert "Error" in result
    mock_mm.save_message.assert_not_called()


def test_invalid_type_returns_error(memorize, mock_mm):
    result = memorize(content="data", type="unknown_type")
    assert "Error" in result
    assert "episodic" in result or "semantic" in result
    mock_mm.save_message.assert_not_called()


def test_wrong_case_type_rejected(memorize, mock_mm):
    result = memorize(content="data", type="Episodic")
    assert "Error" in result
    mock_mm.save_message.assert_not_called()


# ---------------------------------------------------------------------------
# Error recovery
# ---------------------------------------------------------------------------

def test_save_message_raises_returns_error_string(memorize, mock_mm):
    mock_mm.save_message.side_effect = OSError("disk full")
    result = memorize(content="hello", type="semantic")
    assert "Error" in result
    # Tool must not raise — it catches and returns string
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Content integrity
# ---------------------------------------------------------------------------

def test_unicode_content_passed_through(memorize, mock_mm):
    content = "emoji 🎉 and Chinese 中文"
    memorize(content=content, type="semantic")
    saved_content = mock_mm.save_message.call_args[0][1]
    assert saved_content == content


def test_very_long_content_not_truncated(memorize, mock_mm):
    content = "x" * 10_000
    memorize(content=content, type="episodic")
    saved_content = mock_mm.save_message.call_args[0][1]
    assert len(saved_content) == 10_000


def test_content_is_stripped(memorize, mock_mm):
    memorize(content="  hello  ", type="semantic")
    saved_content = mock_mm.save_message.call_args[0][1]
    assert saved_content == "hello"
