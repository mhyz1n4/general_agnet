"""
Top-level pytest configuration and shared fixtures.

Redis setup
-----------
Integration tests that touch Redis expect a Redis instance on port 6380.
Start it with:  make test-infra-start
Stop it with:   make test-infra-stop

The ``clean_redis`` fixture yields a connected client; it skips the test
automatically when port 6380 is not reachable.

LLM setup
---------
LLM integration tests expect a running vLLM (or OpenAI-compatible) endpoint.
The URL is taken from LLM_API_ENDPOINT in .env (default: http://localhost:8000/v1).

The ``llm_available`` fixture skips tests when the endpoint is unreachable.
"""

import json
import os
import time
from urllib import request as urllib_request

import pytest

REDIS_TEST_PORT = 6380


# ---------------------------------------------------------------------------
# Redis fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def redis_client_session():
    """
    Session-scoped Redis client on port 6380.

    Skips the entire session if Redis is not reachable, so individual tests
    do not need to guard themselves.  Start Redis with ``make test-infra-start``.
    """
    try:
        import redis as redis_lib
        client = redis_lib.Redis(
            host="localhost", port=REDIS_TEST_PORT, decode_responses=True
        )
        client.ping()
        return client
    except Exception:
        pytest.skip(
            f"Redis not available on localhost:{REDIS_TEST_PORT}. "
            "Run 'make test-infra-start' first."
        )


@pytest.fixture
def clean_redis(redis_client_session):
    """Per-test Redis client — flushes all keys before and after each test."""
    redis_client_session.flushall()
    yield redis_client_session
    redis_client_session.flushall()


# ---------------------------------------------------------------------------
# LLM availability fixture
# ---------------------------------------------------------------------------

def _llm_endpoint() -> str:
    """Return LLM_API_ENDPOINT from .env or the default vLLM URL."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line.startswith("LLM_API_ENDPOINT="):
                return line.split("=", 1)[1].split("#")[0].strip()
    return os.environ.get("LLM_API_ENDPOINT", "http://localhost:8000/v1")


@pytest.fixture(scope="session")
def llm_available():
    """
    Skip tests that require a live LLM when the endpoint is unreachable.

    Performs a single HTTP GET to /health at session start; subsequent tests
    that depend on this fixture are collected and skipped together if it fails.
    """
    endpoint = _llm_endpoint()
    health_url = endpoint.rstrip("/v1").rstrip("/") + "/health"
    try:
        req = urllib_request.Request(health_url)
        with urllib_request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return endpoint
    except Exception:
        pass
    pytest.skip(
        f"LLM endpoint not reachable at {health_url}. "
        "Start the vLLM server before running LLM integration tests."
    )


@pytest.fixture(scope="session")
def llm_endpoint_or_none():
    """
    Return the LLM API endpoint URL if the server is reachable, else ``None``.

    Unlike ``llm_available``, this fixture never skips — callers decide whether
    to use a real agent or fall back to a mock.
    """
    endpoint = _llm_endpoint()
    health_url = endpoint.rstrip("/v1").rstrip("/") + "/health"
    try:
        req = urllib_request.Request(health_url)
        with urllib_request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return endpoint
    except Exception:
        pass
    return None

