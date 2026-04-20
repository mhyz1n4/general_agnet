"""
Shared fixtures for the V1.1 eval harness.

Provides a fresh ``Orchestrator`` (wired to a real Strands ``Agent`` and a
``StubMemoryProvider``) per fixture, plus a YAML loader.  Skips the entire
module when no LLM endpoint is reachable so the rest of the suite keeps
running.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.agents.context import AgentStateContext
from src.config import Config
from src.hooks.post_session import PostSessionHook
from src.hooks.tool_budget import ToolBudgetHookProvider
from src.memory.stub_provider import StubMemoryProvider
from src.orchestrator import Orchestrator
from tests.eval import load_fixtures


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``eval`` marker so pytest does not warn about it."""
    config.addinivalue_line(
        "markers", "eval: end-to-end golden-fixture tests against a live LLM"
    )


@pytest.fixture(scope="session")
def eval_fixtures() -> List[Dict[str, Any]]:
    """All fixtures parsed from ``seed.yaml``; shared across the session."""
    return load_fixtures()


def _build_agent(endpoint: str, config: Config):
    """
    Construct a minimal Strands Agent wired to the live LLM endpoint.

    Budget enforcement is installed via ``ToolBudgetHookProvider`` and a
    fresh ``AgentStateContext`` attached to the agent — the Orchestrator
    will leave it in place and reset it per turn.
    """
    from strands import Agent
    from strands.models.openai import OpenAIModel

    model = OpenAIModel(
        client_args={"api_key": config.llm_api_key, "base_url": endpoint},
        model_id=config.llm_model,
        params={"max_tokens": max(config.llm_max_tokens, 4096)},
    )
    agent = Agent(
        model=model,
        tools=[],
        callback_handler=None,
        hooks=[ToolBudgetHookProvider()],
    )
    agent.state_context = AgentStateContext(max_tool_calls=config.max_tool_calls)
    return agent


@pytest.fixture
def eval_orchestrator(llm_endpoint_or_none, tmp_path):
    """
    Factory fixture: returns a callable that builds a fresh orchestrator per
    fixture, optionally preseeding the stub memory provider with a list of
    ``{content, type, topic}`` dicts.

    Skips the test when no LLM endpoint is reachable — eval only has teeth
    when run against a real model.
    """
    if llm_endpoint_or_none is None:
        pytest.skip("LLM endpoint unreachable — eval harness requires a live model")

    def _make(seed_memory: List[Dict[str, Any]] | None = None) -> Orchestrator:
        config = Config()
        provider = StubMemoryProvider()
        for rec in seed_memory or []:
            provider.save(
                content=rec["content"],
                type=rec["type"],
                topic=rec.get("topic"),
            )

        agent = _build_agent(llm_endpoint_or_none, config)

        cfg = MagicMock(spec=Config)
        cfg.session_inactivity_timeout_seconds = config.session_inactivity_timeout_seconds
        cfg.tool_timeout_seconds = config.tool_timeout_seconds
        cfg.max_context_chars = config.max_context_chars
        cfg.max_tool_calls = config.max_tool_calls
        cfg.llm_max_tokens = config.llm_max_tokens
        cfg.memory_root = str(tmp_path / "memory")

        return Orchestrator(
            agent=agent,
            memory_provider=provider,
            post_session_hook=MagicMock(spec=PostSessionHook),
            config=cfg,
            session_id="eval_session",
        )

    return _make
