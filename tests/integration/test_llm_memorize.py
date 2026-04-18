"""
LLM integration tests for the memorize tool call flow.

V1.1: Uses MemoryProvider (StubMemoryProvider) instead of MemoryManager.
Tests verify the LLM calls memorize() with correct types and the provider
stores the entries.

Requirements:
  - A running vLLM endpoint (default: http://localhost:8000/v1).
  - The endpoint must use ``--tool-call-parser hermes`` (or equivalent).

Run:
    make test-llm
    pytest tests/integration/test_llm_memorize.py -v -m llm -s
"""

import re
import uuid

import pytest

from src.config import Config
from src.constants import SESSION_ID_HEX_LENGTH
from src.memory.stub_provider import StubMemoryProvider
from src.memory.provider import SearchFilters
from src.prompts.loader import render_prompt
from src.tools.memorize import create_memorize_tool

pytestmark = pytest.mark.llm


# ---------------------------------------------------------------------------
# Agent fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_ctx(llm_available, session_id, strands_patch):
    """
    Build a real Strands Agent wired to a StubMemoryProvider.

    Requires ``llm_available`` — this fixture (and every test that depends on
    it) is automatically skipped when the vLLM endpoint is unreachable.
    """
    try:
        from strands import Agent
        from strands.models.openai import OpenAIModel
    except ImportError:
        pytest.skip("strands-agents package not installed")

    from src.orchestrator import tool_call_counter

    config = Config()
    provider = StubMemoryProvider()
    memorize_tool = create_memorize_tool(provider, session_id=session_id)
    system_prompt = render_prompt("system_prompt")

    model = OpenAIModel(
        client_args={
            "api_key": config.llm_api_key,
            "base_url": config.llm_api_endpoint,
        },
        model_id=config.llm_model,
        params={"max_tokens": config.llm_max_tokens},
    )
    agent = Agent(
        model=model,
        tools=[memorize_tool],
        system_prompt=system_prompt,
        callback_handler=None,
    )

    tool_call_counter.count = 0
    tool_call_counter.limit = config.max_tool_calls

    return {
        "agent": agent,
        "provider": provider,
        "session_id": session_id,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINKING_RE = re.compile(
    r"<think(?:ing)?>\s*.*?\s*</think(?:ing)?>", re.DOTALL | re.IGNORECASE
)


def _call_agent(agent_ctx: dict, user_input: str) -> str:
    """Call the agent and return the cleaned response."""
    from src.orchestrator import tool_call_counter

    tool_call_counter.count = 0
    tool_call_counter.limit = agent_ctx["config"].max_tool_calls

    response = agent_ctx["agent"](user_input)
    return _THINKING_RE.sub("", str(response)).strip()


# ---------------------------------------------------------------------------
# Test 1 — Episodic memory tool call
# ---------------------------------------------------------------------------

def test_episodic_tool_call(agent_ctx):
    """The LLM should call memorize(type='episodic') for a personal event."""
    provider = agent_ctx["provider"]

    _call_agent(agent_ctx, "memorize this: I went for a run in the park this morning")

    items = [i for i in provider._items.values() if i.type == "episodic"]
    assert len(items) >= 1, (
        f"Expected at least one episodic item, got: {list(provider._items.values())}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Semantic memory tool call
# ---------------------------------------------------------------------------

def test_semantic_tool_call(agent_ctx):
    """The LLM should call memorize(type='semantic') for a persistent fact."""
    provider = agent_ctx["provider"]

    _call_agent(agent_ctx, "remember that my favorite programming language is Python")

    items = [i for i in provider._items.values() if i.type == "semantic"]
    assert len(items) >= 1, (
        f"Expected at least one semantic item, got: {list(provider._items.values())}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Procedural memory tool call
# ---------------------------------------------------------------------------

def test_procedural_tool_call(agent_ctx):
    """The LLM should call memorize(type='procedural') for a how-to."""
    provider = agent_ctx["provider"]

    _call_agent(
        agent_ctx,
        "save this procedure: to check GPU usage on this machine, run `nvidia-smi`",
    )

    items = [i for i in provider._items.values() if i.type == "procedural"]
    assert len(items) >= 1, (
        f"Expected at least one procedural item, got: {list(provider._items.values())}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Memory retrieval via orchestrator
# ---------------------------------------------------------------------------

def test_memory_retrieval(agent_ctx):
    """
    Pre-seed the memory with a known fact, then ask the agent about it.
    The agent must incorporate the fact in its response.
    """
    from src.hooks.post_session import PostSessionHook
    from src.orchestrator import Orchestrator, tool_call_counter
    from rich.console import Console
    import os

    provider = agent_ctx["provider"]
    config = agent_ctx["config"]
    session_id = agent_ctx["session_id"]

    provider.save(
        "Sarah's birthday is on July 22nd.",
        type="semantic",
        topic="people",
    )

    post_hook = PostSessionHook(
        session_id=session_id,
        metrics_path="/tmp/test_metrics.jsonl",
    )
    orchestrator = Orchestrator(
        agent=agent_ctx["agent"],
        memory_provider=provider,
        post_session_hook=post_hook,
        config=config,
        session_id=session_id,
        console=Console(quiet=True),
    )

    tool_call_counter.count = 0
    tool_call_counter.limit = config.max_tool_calls

    response = orchestrator._process_turn("When is Sarah's birthday?")

    assert any(term in response.lower() for term in ("july", "22", "sarah")), (
        f"Expected response to mention Sarah's birthday (July 22nd), got: {response!r}"
    )
