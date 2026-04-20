"""
Research sub-agent.

Exposes a single Strands tool, ``delegate_to_research``, that hands a focused
question to a dedicated sub-agent equipped with ``web_search`` and
``search_memory``.  The sub-agent runs in its own worker thread with an
independent ``AgentStateContext`` (so its tool-call budget never debits the
parent's) and a wall-clock timeout that is *not* defeated by executor
shutdown:

  - The executor is module-scoped and never joined.  A ``with``-managed
    executor would call ``shutdown(wait=True)`` on exit, blocking the parent
    until a stuck sub-agent finishes — silently neutralising the timeout.
  - On timeout we ``future.cancel()`` and abandon the worker; the parent
    returns the timeout message immediately.

Sub-agent isolation rests on the per-agent ``state_context``: the
``enforce_tool_budget`` hook reads ``event.agent.state_context``, so the
correct context is always visible regardless of which thread is executing.
"""

from __future__ import annotations

import concurrent.futures

from strands import tool

from src.agents.context import AgentStateContext
from src.config import Config
from src.constants import (
    DEFAULT_TOOL_BUDGET_WEB_SEARCH,
    QUERY_LOG_PREVIEW_LENGTH,
)
from src.hooks.tool_budget import ToolBudgetHookProvider
from src.logging_config import get_logger
from src.memory.provider import MemoryProvider
from src.prompts.loader import render_prompt
from src.tools.search_memory import create_search_memory_tool
from src.tools.web_search import create_web_search_tool_from_config

logger = get_logger(__name__)

DELEGATE_TOOL_NAME = "delegate_to_research"
DELEGATE_TOOL_DESCRIPTION = (
    "Delegate a focused research question to a sub-agent that can call "
    "web_search and search_memory. Pass the full user question as ``query``. "
    "Use this for multi-step research, current-events lookups, or any query "
    "needing more than one tool call. Not for simple chit-chat."
)

# Single long-lived worker pool shared across all delegations.  Using a
# context-managed executor would defeat the wall-clock timeout: ``__exit__``
# calls ``shutdown(wait=True)`` and would block the parent until the stuck
# sub-agent returns.  ``max_workers=1`` matches the documented "one
# delegation at a time" semantic; bump if delegations ever run concurrently.
_DELEGATION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="research-delegation"
)


def _build_sub_agent(memory_provider: MemoryProvider, config: Config):
    """
    Construct the research sub-agent, or return ``None`` when Strands is
    missing.  The sub-agent carries its own ``AgentStateContext`` so its
    tool calls are budgeted independently from the parent's.
    """
    try:
        from strands import Agent
        from strands.models.openai import OpenAIModel
    except ImportError:
        logger.warning("research sub-agent: strands not importable; skipping")
        return None

    search_memory_tool = create_search_memory_tool(memory_provider)
    web_search_tool = create_web_search_tool_from_config(config)

    sub_tools = [search_memory_tool]
    if web_search_tool is not None:
        sub_tools.append(web_search_tool)

    model = OpenAIModel(
        client_args={
            "api_key": config.llm_api_key,
            "base_url": config.llm_api_endpoint,
        },
        model_id=config.llm_model,
        params={"max_tokens": max(config.llm_max_tokens, 4096)},
    )

    system_prompt = render_prompt("research_agent")

    agent = Agent(
        name="research",
        model=model,
        tools=sub_tools,
        system_prompt=system_prompt,
        callback_handler=None,
        hooks=[ToolBudgetHookProvider()],
    )

    agent.state_context = AgentStateContext(
        max_tool_calls=config.sub_agent_max_tool_calls,
        per_tool_limits={
            "web_search": getattr(
                config,
                "tool_budget_web_search",
                DEFAULT_TOOL_BUDGET_WEB_SEARCH,
            ),
        },
    )
    return agent


def create_research_agent_tool(
    memory_provider: MemoryProvider,
    config: Config,
):
    """
    Build the research sub-agent and return it wrapped as a Strands ``@tool``.

    Returns:
        The wrapped ``delegate_to_research`` tool, or ``None`` if the
        sub-agent cannot be constructed (e.g. Strands is unavailable).
    """
    sub_agent = _build_sub_agent(memory_provider, config)
    if sub_agent is None:
        return None

    timeout_s = config.sub_agent_timeout_seconds

    @tool(name=DELEGATE_TOOL_NAME, description=DELEGATE_TOOL_DESCRIPTION)
    def delegate_to_research(query: str) -> str:
        """Hand *query* to the research sub-agent and return its answer."""
        if not query or not query.strip():
            return "delegate_to_research: query cannot be empty"

        sub_agent.state_context.reset()

        future = _DELEGATION_EXECUTOR.submit(lambda: str(sub_agent(query)))
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            # Cancel the bookkeeping; the worker thread itself cannot be
            # killed but is abandoned — the next delegation will queue
            # behind it on the single-worker pool.  That's acceptable: the
            # alternative (joining here) blocks the parent agent's loop.
            future.cancel()
            logger.warning(
                "delegate_to_research: sub-agent timed out",
                extra={"data": {"timeout_s": timeout_s}},
            )
            return (
                f"delegate_to_research: sub-agent exceeded "
                f"{timeout_s}s and was cancelled"
            )
        except Exception as exc:
            logger.error(
                "delegate_to_research: sub-agent failed",
                extra={"data": {"error": str(exc)}},
            )
            return f"delegate_to_research: sub-agent error: {exc}"

    return delegate_to_research
