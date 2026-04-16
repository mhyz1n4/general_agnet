"""
LLM integration tests for the memorize tool call flow.

These tests use a real LLM endpoint (vLLM or OpenAI-compatible) to verify:

  1. The LLM calls memorize() with the correct memory type for each category:
       - episodic  → personal events / things that happened
       - semantic  → facts, preferences, knowledge
       - procedural → how-to instructions / workflows

  2. Each memory type is stored in the correct filesystem directory:
       - episodic   → memory/conversations/{session_id}/{YYYY-MM-DD}/
       - semantic   → memory/knowledge/{topic}/
       - procedural → memory/procedures/{topic}/

  3. The index.json is updated with a key matching the memory type.

  4. The agent retrieves previously saved memories when queried and
     incorporates them into its response.

Requirements:
  - A running vLLM endpoint (default: http://localhost:8000/v1).
    Set LLM_API_ENDPOINT / LLM_MODEL / LLM_API_KEY in .env to configure.
  - The endpoint must use ``--tool-call-parser hermes`` (or equivalent) to
    return properly structured tool_calls in the API response.

Run:
    make test-llm
    pytest tests/integration/test_llm_memorize.py -v -m llm -s
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.config import Config
from src.constants import (
    MEMORY_TIMESTAMP_DISPLAY_FORMAT,
    SESSION_ID_HEX_LENGTH,
)
from src.prompts.loader import render_prompt
from src.tools.memorize import create_memorize_tool

pytestmark = pytest.mark.llm

# session_id, mem_stack, and strands_patch come from tests/integration/conftest.py


# ---------------------------------------------------------------------------
# Agent fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_ctx(llm_available, mem_stack, session_id, strands_patch):
    """
    Build a real Strands Agent wired to the test memory stack.

    Requires ``llm_available`` — this fixture (and every test that depends on
    it) is automatically skipped when the vLLM endpoint is unreachable.

    The ``strands_patch`` fixture is applied first to handle vLLM streaming
    quirks (strands issue #815) via ``unittest.mock.patch`` semantics.

    Returns a dict with:
      ``agent``      — callable Strands Agent with the memorize tool
      ``mem_stack``  — isolated memory stack dict (root, manager, spy_calls, …)
      ``session_id`` — session ID used for episodic directory scoping
      ``config``     — ``Config`` instance loaded from .env / YAML
    """
    try:
        from strands import Agent
        from strands.models.openai import OpenAIModel
    except ImportError:
        pytest.skip("strands-agents package not installed")

    from src.orchestrator import tool_call_counter

    config = Config()
    memorize_tool = create_memorize_tool(mem_stack["manager"], session_id=session_id)
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

    # Reset the per-invocation tool call counter
    tool_call_counter.count = 0
    tool_call_counter.limit = config.max_tool_calls

    return {
        "agent": agent,
        "mem_stack": mem_stack,
        "session_id": session_id,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINKING_RE = re.compile(
    r"<think(?:ing)?>\s*.*?\s*</think(?:ing)?>", re.DOTALL | re.IGNORECASE
)


def _md_files_under(root: str, subdir: str) -> list[str]:
    """Return absolute paths of all ``.md`` files under ``root/subdir``."""
    target = os.path.join(root, subdir)
    if not os.path.isdir(target):
        return []
    result = []
    for dirpath, _, filenames in os.walk(target):
        for fname in filenames:
            if fname.endswith(".md"):
                result.append(os.path.join(dirpath, fname))
    return result


def _index_keys_of_type(index_path: str, mem_type: str) -> list[str]:
    """Return keys in ``index.json`` whose prefix matches ``mem_type``."""
    if not os.path.exists(index_path):
        return []
    with open(index_path) as f:
        index = json.load(f)
    return [k for k in index if k.startswith(f"{mem_type}_")]


def _memorize_spy_calls(spy_calls: list[dict], mem_type: str) -> list[dict]:
    """Filter ``spy_calls`` to only entries with the given memory type."""
    return [c for c in spy_calls if c["metadata"].get("type") == mem_type]


def _call_agent(agent_ctx: dict, user_input: str) -> str:
    """
    Call the agent with ``user_input`` and return the cleaned response.

    Resets the tool call counter before each invocation and strips any
    ``<think>…</think>`` blocks from the raw response.
    """
    from src.orchestrator import tool_call_counter

    tool_call_counter.count = 0
    tool_call_counter.limit = agent_ctx["config"].max_tool_calls

    response = agent_ctx["agent"](user_input)
    return _THINKING_RE.sub("", str(response)).strip()


# ---------------------------------------------------------------------------
# Test 1 — Episodic memory tool call
# ---------------------------------------------------------------------------

def test_episodic_tool_call(agent_ctx):
    """
    The LLM should call ``memorize(type='episodic')`` for a personal event.
    The resulting ``.md`` file must appear under
    ``memory/conversations/{session_id}/{date}/``.
    """
    mem_stack = agent_ctx["mem_stack"]
    session_id = agent_ctx["session_id"]

    _call_agent(agent_ctx, "memorize this: I went for a run in the park this morning")

    # Tool was called with type=episodic
    episodic_calls = _memorize_spy_calls(mem_stack["spy_calls"], "episodic")
    assert len(episodic_calls) >= 1, (
        f"Expected at least one episodic memorize call, got: {mem_stack['spy_calls']}"
    )

    # Content is timestamp-enriched
    content = episodic_calls[0]["content"]
    assert re.match(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\]", content), (
        f"Expected timestamp prefix in content, got: {content!r}"
    )

    # File written to conversations/{session_id}/{date}/
    md_files = _md_files_under(mem_stack["root"], "conversations")
    assert len(md_files) >= 1, (
        f"Expected .md files under conversations/, found none. "
        f"Memory root contents: {os.listdir(mem_stack['root'])}"
    )
    assert any(session_id in p for p in md_files), (
        f"Expected file under session folder {session_id!r}, got: {md_files}"
    )

    # Index updated with an episodic key
    index_keys = _index_keys_of_type(mem_stack["index_path"], "episodic")
    assert len(index_keys) >= 1, "Expected at least one episodic key in index.json"


# ---------------------------------------------------------------------------
# Test 2 — Semantic memory tool call
# ---------------------------------------------------------------------------

def test_semantic_tool_call(agent_ctx):
    """
    The LLM should call ``memorize(type='semantic')`` for a persistent fact or
    preference.  The ``.md`` file should appear under ``memory/knowledge/``.
    """
    mem_stack = agent_ctx["mem_stack"]

    _call_agent(agent_ctx, "remember that my favorite programming language is Python")

    # Tool was called with type=semantic
    semantic_calls = _memorize_spy_calls(mem_stack["spy_calls"], "semantic")
    assert len(semantic_calls) >= 1, (
        f"Expected at least one semantic memorize call, got: {mem_stack['spy_calls']}"
    )

    # Content is timestamp-enriched
    content = semantic_calls[0]["content"]
    assert re.match(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\]", content), (
        f"Expected timestamp prefix in content, got: {content!r}"
    )

    # File written to knowledge/
    md_files = _md_files_under(mem_stack["root"], "knowledge")
    assert len(md_files) >= 1, (
        f"Expected .md files under knowledge/, found none. "
        f"Memory root: {os.listdir(mem_stack['root'])}"
    )

    # Index updated with a semantic key
    index_keys = _index_keys_of_type(mem_stack["index_path"], "semantic")
    assert len(index_keys) >= 1, "Expected at least one semantic key in index.json"


# ---------------------------------------------------------------------------
# Test 3 — Procedural memory tool call
# ---------------------------------------------------------------------------

def test_procedural_tool_call(agent_ctx):
    """
    The LLM should call ``memorize(type='procedural')`` for a how-to or
    workflow.  The ``.md`` file should appear under ``memory/procedures/``.
    """
    mem_stack = agent_ctx["mem_stack"]

    _call_agent(
        agent_ctx,
        "save this procedure: to check GPU usage on this machine, run `nvidia-smi`",
    )

    # Tool was called with type=procedural
    procedural_calls = _memorize_spy_calls(mem_stack["spy_calls"], "procedural")
    assert len(procedural_calls) >= 1, (
        f"Expected at least one procedural memorize call, got: {mem_stack['spy_calls']}"
    )

    # Content is timestamp-enriched
    content = procedural_calls[0]["content"]
    assert re.match(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\]", content), (
        f"Expected timestamp prefix in content, got: {content!r}"
    )

    # File written to procedures/
    md_files = _md_files_under(mem_stack["root"], "procedures")
    assert len(md_files) >= 1, (
        f"Expected .md files under procedures/, found none. "
        f"Memory root: {os.listdir(mem_stack['root'])}"
    )

    # Index updated with a procedural key
    index_keys = _index_keys_of_type(mem_stack["index_path"], "procedural")
    assert len(index_keys) >= 1, "Expected at least one procedural key in index.json"


# ---------------------------------------------------------------------------
# Test 4 — Memory retrieval: agent answers from pre-seeded memory
# ---------------------------------------------------------------------------

def test_memory_retrieval(agent_ctx):
    """
    Pre-seed the memory with a known fact, then ask the agent about it.
    The agent must retrieve the fact and include it in its response.

    This tests the full retrieval loop:
      pre_mem_fetch → retriever.search → context prefix injected → LLM response
    """
    from src.hooks.pre_mem_fetch import PreMemFetchHook
    from src.hooks.post_mem_fetch import PostMemFetchHook
    from src.hooks.post_session import PostSessionHook
    from src.orchestrator import Orchestrator, tool_call_counter
    from rich.console import Console

    mem_stack = agent_ctx["mem_stack"]
    config = agent_ctx["config"]
    session_id = agent_ctx["session_id"]
    manager = mem_stack["manager"]

    # Pre-seed a memorable fact using the shared timestamp format
    now = datetime.now(timezone.utc)
    manager.save_message(
        "semantic_birthday_fact",
        f"[{now.strftime(MEMORY_TIMESTAMP_DISPLAY_FORMAT)}] Sarah's birthday is on July 22nd.",
        {"type": "semantic", "topic": "people", "timestamp": now.isoformat()},
    )

    # Wire up an Orchestrator with the real agent
    post_hook = PostSessionHook(
        memory_manager=manager,
        session_id=session_id,
        metrics_path=os.path.join(mem_stack["root"], "metrics.jsonl"),
    )
    orchestrator = Orchestrator(
        agent=agent_ctx["agent"],
        memory_manager=manager,
        pre_mem_fetch_hook=PreMemFetchHook(),
        post_mem_fetch_hook=PostMemFetchHook(max_context_chars=config.max_context_chars),
        post_session_hook=post_hook,
        config=config,
        session_id=session_id,
        console=Console(quiet=True),
    )

    # Reset counter before the orchestrated turn
    tool_call_counter.count = 0
    tool_call_counter.limit = config.max_tool_calls

    response = orchestrator._process_turn("When is Sarah's birthday?")

    # The response should mention July or the 22nd
    assert any(term in response.lower() for term in ("july", "22", "sarah")), (
        f"Expected response to mention Sarah's birthday (July 22nd), got: {response!r}"
    )
