"""
Unit tests for the ``ToolResult`` envelope helpers (V1.1 M1).
"""

from __future__ import annotations

import pytest

from src.tools.envelope import ToolResult, err, ok


def test_ok_default_data_is_none() -> None:
    """``ok()`` with no argument yields a successful envelope carrying None."""
    result = ok()
    assert result == ToolResult(ok=True, data=None, error=None, metadata={})


def test_ok_carries_data_and_metadata() -> None:
    """``ok(data, **meta)`` populates data and metadata, leaving error None."""
    result = ok({"key": "abc"}, source="memorize", count=1)
    assert result["ok"] is True
    assert result["data"] == {"key": "abc"}
    assert result["metadata"] == {"source": "memorize", "count": 1}
    assert result["error"] is None


def test_err_populates_error_and_nulls_data() -> None:
    """``err(msg, **meta)`` sets ok=False with no payload and the message."""
    result = err("bad input", field="type")
    assert result["ok"] is False
    assert result["data"] is None
    assert result["error"] == "bad input"
    assert result["metadata"] == {"field": "type"}


def test_metadata_is_always_a_dict() -> None:
    """``metadata`` defaults to an empty dict even when no kwargs are given."""
    assert ok()["metadata"] == {}
    assert err("boom")["metadata"] == {}


def test_metadata_does_not_leak_between_calls() -> None:
    """Each call gets a fresh metadata dict so mutating one cannot affect another."""
    a = ok()
    a["metadata"]["dirty"] = True
    b = ok()
    assert b["metadata"] == {}
