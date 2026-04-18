"""
Web search tool (V1.1 M2) — thin Tavily client exposed as a Strands @tool.

Design notes
------------
- Synchronous HTTP via ``urllib.request`` keeps the tool dependency-free and
  sidesteps the async/sync impedance mismatch between ``aiohttp`` and the
  Strands tool interface.
- Single retry on 5xx; 4xx surfaces as a failure envelope without retry.
- Factory pattern mirrors ``create_memorize_tool``: returns ``None`` when no
  API token is configured so ``main.py`` can simply skip registering the tool.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from strands import tool

from src.constants import (
    DEFAULT_TAVILY_ENDPOINT,
    DEFAULT_TAVILY_MAX_RESULTS,
    DEFAULT_TAVILY_TIMEOUT_SECONDS,
)
from src.logging_config import get_logger
from src.tools.envelope import ToolResult, err, ok

logger = get_logger(__name__)

_MAX_RESULTS_HARD_CAP = 20
_RETRYABLE_STATUS = {500, 502, 503, 504}


def _post_json(
    url: str, payload: Dict[str, Any], timeout: int
) -> tuple[int, Dict[str, Any]]:
    """
    POST ``payload`` as JSON to ``url`` and return ``(status, body_dict)``.

    Raises ``urllib.error.URLError`` on transport failure.  On HTTP error
    responses the status code is returned alongside the decoded body.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body
    except urllib.error.HTTPError as http_err:
        body_text = http_err.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            body = {"message": body_text}
        return http_err.code, body


def create_web_search_tool(
    api_token: str,
    endpoint: str = DEFAULT_TAVILY_ENDPOINT,
    timeout_seconds: int = DEFAULT_TAVILY_TIMEOUT_SECONDS,
):
    """
    Create a Strands @tool that performs a Tavily web search.

    Args:
        api_token:        Tavily API key.
        endpoint:         Tavily search endpoint URL.
        timeout_seconds:  Per-request wall-clock timeout.

    Returns:
        A Strands tool function that returns a ``ToolResult`` envelope.  The
        ``data`` field on success is a list of result dicts; on failure the
        ``error`` field carries a human-readable message.
    """

    @tool
    def web_search(query: str, max_results: int = DEFAULT_TAVILY_MAX_RESULTS) -> ToolResult:
        """
        Search the public web for recent, relevant information on ``query``.

        Args:
            query:        Search terms in natural language.
            max_results:  Upper bound on the number of results returned.

        Returns:
            ``ToolResult`` envelope. On success ``data`` is the list of
            result dicts from Tavily; on failure ``error`` carries the reason.
        """
        if not query.strip():
            return err("query cannot be empty")

        clamped = max(1, min(int(max_results), _MAX_RESULTS_HARD_CAP))
        payload = {
            "api_key": api_token,
            "query": query.strip(),
            "max_results": clamped,
        }

        for attempt in range(2):
            try:
                status, body = _post_json(endpoint, payload, timeout_seconds)
            except urllib.error.URLError as net_err:
                logger.warning(
                    "web_search: transport error",
                    extra={"data": {"error": str(net_err), "attempt": attempt}},
                )
                if attempt == 0:
                    continue
                return err(f"network error: {net_err}")

            if status in _RETRYABLE_STATUS and attempt == 0:
                logger.warning(
                    "web_search: retryable status",
                    extra={"data": {"status": status}},
                )
                continue
            if 400 <= status < 600:
                message = body.get("message") or body.get("error") or f"HTTP {status}"
                return err(f"search failed: {message}", status=status)

            results: List[Dict[str, Any]] = list(body.get("results") or [])
            return ok(
                results[:clamped],
                query=payload["query"],
                result_count=len(results[:clamped]),
            )

        return err("search failed after retry")

    return web_search


def create_web_search_tool_from_config(config) -> Optional[Any]:
    """
    Wire a ``web_search`` tool from a ``Config`` instance, or return ``None``.

    Returning ``None`` when the token is absent lets ``main.py`` conditionally
    register the tool without needing to know the factory's internals.
    """
    if not getattr(config, "tavily_search_token", None):
        logger.info("web_search: skipping registration (no TAVILY_SEARCH_TOKEN)")
        return None
    return create_web_search_tool(
        api_token=config.tavily_search_token,
        endpoint=config.tavily_search_endpoint,
        timeout_seconds=config.tavily_search_timeout_seconds,
    )
