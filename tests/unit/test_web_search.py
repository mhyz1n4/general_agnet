"""
Unit tests for the ``web_search`` tool (V1.1 M2).

Network layer is mocked — we stub ``_post_json`` so the tests stay offline and
deterministic. These assert on the ``ToolResult`` envelope, retry behaviour
on 5xx, fast-fail on 4xx, and the config-based factory.
"""

from __future__ import annotations

import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.tools import web_search as ws


@pytest.fixture
def api_token() -> str:
    """Arbitrary non-empty token; never leaves the test."""
    return "test-token"


@pytest.fixture
def tool_factory(api_token):
    """Callable returning a fresh web_search tool bound to the test token."""
    return lambda **kwargs: ws.create_web_search_tool(api_token, **kwargs)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_search_returns_results_envelope(tool_factory) -> None:
    """A 200 response with ``results`` is wrapped in ok(data=list)."""
    tool = tool_factory()
    body = {"results": [{"title": "foo", "url": "https://example.com"}]}
    with patch.object(ws, "_post_json", return_value=(200, body)) as post:
        result = tool(query="hello world")

    assert result["ok"] is True
    assert result["data"] == body["results"]
    assert result["metadata"]["query"] == "hello world"
    assert result["metadata"]["result_count"] == 1
    post.assert_called_once()


def test_max_results_clamped_to_hard_cap(tool_factory) -> None:
    """``max_results`` is clamped to the internal hard cap before POSTing."""
    tool = tool_factory()
    with patch.object(ws, "_post_json", return_value=(200, {"results": []})) as post:
        tool(query="x", max_results=9999)

    sent_payload = post.call_args[0][1]
    assert sent_payload["max_results"] == ws._MAX_RESULTS_HARD_CAP


def test_max_results_clamped_to_minimum(tool_factory) -> None:
    """``max_results`` of 0 or negative is lifted to 1."""
    tool = tool_factory()
    with patch.object(ws, "_post_json", return_value=(200, {"results": []})) as post:
        tool(query="x", max_results=0)
    assert post.call_args[0][1]["max_results"] == 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_empty_query_short_circuits(tool_factory) -> None:
    """An empty query never hits the network and returns ok=False."""
    tool = tool_factory()
    with patch.object(ws, "_post_json") as post:
        result = tool(query="   ")
    assert result["ok"] is False
    assert "empty" in result["error"]
    post.assert_not_called()


def test_4xx_fails_fast_without_retry(tool_factory) -> None:
    """A 401 response returns err immediately; no retry attempted."""
    tool = tool_factory()
    body = {"message": "invalid token"}
    with patch.object(ws, "_post_json", return_value=(401, body)) as post:
        result = tool(query="hello")

    assert result["ok"] is False
    assert "invalid token" in result["error"]
    assert result["metadata"]["status"] == 401
    assert post.call_count == 1


def test_5xx_retried_once_then_surfaced(tool_factory) -> None:
    """A 503 on first call triggers one retry; second 503 surfaces err."""
    tool = tool_factory()
    with patch.object(ws, "_post_json", return_value=(503, {"message": "down"})) as post:
        result = tool(query="hello")

    assert result["ok"] is False
    assert post.call_count == 2
    assert result["metadata"]["status"] == 503


def test_5xx_then_200_returns_ok(tool_factory) -> None:
    """A 500 first then 200 means the retry succeeded."""
    tool = tool_factory()
    side = [(500, {"message": "oops"}), (200, {"results": [{"title": "ok"}]})]
    with patch.object(ws, "_post_json", side_effect=side) as post:
        result = tool(query="hello")

    assert result["ok"] is True
    assert post.call_count == 2


def test_network_error_retried_once(tool_factory) -> None:
    """A ``URLError`` on first call triggers one retry; second one surfaces err."""
    tool = tool_factory()
    with patch.object(
        ws, "_post_json", side_effect=urllib.error.URLError("DNS failure")
    ) as post:
        result = tool(query="hello")

    assert result["ok"] is False
    assert "network error" in result["error"]
    assert post.call_count == 2


# ---------------------------------------------------------------------------
# Factory from Config
# ---------------------------------------------------------------------------


def test_factory_returns_none_when_token_missing() -> None:
    """No token configured → factory returns None so main.py can skip registering."""
    cfg = SimpleNamespace(
        tavily_search_token=None,
        tavily_search_endpoint="https://api.tavily.com/search",
    )
    assert ws.create_web_search_tool_from_config(cfg) is None


def test_factory_returns_callable_when_token_present() -> None:
    """With a token configured the factory returns a callable tool."""
    cfg = SimpleNamespace(
        tavily_search_token="t",
        tavily_search_endpoint="https://api.tavily.com/search",
    )
    tool = ws.create_web_search_tool_from_config(cfg)
    assert tool is not None
    assert callable(tool)
