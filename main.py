"""
Entry point for the personal assistant agent.

V1.1: Wire MemoryProvider (ReMeLightProvider), run PreSessionHook, then
start the Orchestrator chat loop.  Redis, DLQ, and the V1 memory stack
have been retired.
"""

import os
import sys
import uuid
from datetime import datetime, timezone

from rich.console import Console

from src.agents.context import AgentStateContext
from src.config import Config
from src.logging_config import get_logger, init_session_logging
from src.memory.remelight.compaction_manager import ReMeCompactionManager
from src.memory.remelight.provider import ReMeLightProvider
from src.hooks.pre_session import PreSessionHook
from src.hooks.post_session import PostSessionHook
from src.hooks.tool_budget import ToolBudgetHookProvider
from src.prompts.loader import render_prompt
from src.agents.research import create_research_agent_tool
from src.tools.confirm import set_confirmation_mode
from src.tools.memorize import create_memorize_tool
from src.tools.run_python import create_run_python_tool_from_config
from src.tools.search_memory import create_search_memory_tool
from src.tools.web_search import create_web_search_tool_from_config
from src.constants import (
    DEFAULT_METRICS_FILENAME,
    DEFAULT_TOOL_BUDGET_DELEGATE_TO_RESEARCH,
    DEFAULT_TOOL_BUDGET_RUN_PYTHON,
    DEFAULT_TOOL_BUDGET_WEB_SEARCH,
    SESSION_ID_HEX_LENGTH,
    SESSION_LOG_DATE_FORMAT,
    SESSION_LOG_SUBDIR_PREFIX,
)
from src.orchestrator import Orchestrator

console = Console()


def main() -> None:
    """Wire all components and run the chat session."""
    config = Config()

    # Build session ID first so the log directory can include it.
    session_id = uuid.uuid4().hex[:SESSION_ID_HEX_LENGTH]
    session_date = datetime.now(timezone.utc).strftime(SESSION_LOG_DATE_FORMAT)
    session_log_dir = os.path.join(
        config.log_dir,
        f"{SESSION_LOG_SUBDIR_PREFIX}{session_id}_{session_date}",
    )
    init_session_logging(
        session_log_dir,
        log_level=config.log_level,
        strands_log_level=config.strands_log_level,
    )
    logger = get_logger(__name__)

    logger.info("main: starting session", extra={"data": {"session_id": session_id}})

    # Propagate the configured confirmation mode to the ContextVar so every
    # @requires_confirmation-decorated tool picks it up.
    set_confirmation_mode(config.tool_confirmation_mode)  # type: ignore[arg-type]

    # --- Memory provider (ReMeLight) ---
    memory_provider = ReMeLightProvider(
        memory_root=config.memory_root,
        session_id=session_id,
    )

    # --- LLM client (OpenAI-compatible; points to local vLLM by default) ---
    from src.client.open_ai_model import OpenAIModelClient

    llm_client = OpenAIModelClient(
        api_endpoint=config.llm_api_endpoint,
        token=config.llm_api_key,
    )

    # --- Pre-session hook ---
    pre_session_hook = PreSessionHook(
        memory_root=config.memory_root,
        llm_client=llm_client,
        llm_model=config.llm_model,
        log_dir=config.log_dir,
    )

    pre_result = pre_session_hook.run()
    if not pre_result.success:
        console.print(f"[bold red]Startup failed:[/bold red] {pre_result.message}")
        sys.exit(1)

    if pre_result.degraded:
        console.print(f"[yellow]Warning:[/yellow] {pre_result.message}")

    # --- Strands Agent ---
    try:
        from strands import Agent
        from strands.models.openai import OpenAIModel
    except ImportError:
        console.print("[red]strands-agents package not installed. Run: pip install strands-agents[/red]")
        sys.exit(1)

    memorize_tool = create_memorize_tool(memory_provider, session_id=session_id)
    search_memory_tool = create_search_memory_tool(memory_provider)
    web_search_tool = create_web_search_tool_from_config(config)
    research_tool = create_research_agent_tool(memory_provider, config)
    run_python_tool = create_run_python_tool_from_config(config)
    system_prompt = render_prompt("system_prompt")

    agent_tools = [memorize_tool, search_memory_tool]
    if web_search_tool is not None:
        agent_tools.append(web_search_tool)
    if research_tool is not None:
        agent_tools.append(research_tool)
    if run_python_tool is not None:
        agent_tools.append(run_python_tool)

    # Prompt caching is transparent on OpenAI-compatible backends (OpenAI,
    # vLLM with --enable-prefix-caching).  On Anthropic-compatible proxies
    # Strands surfaces cacheReadInputTokens via
    # ``agent.event_loop_metrics.accumulated_usage``; the Orchestrator reads
    # it into ``SessionMetrics.cache_read_input_tokens`` at end-of-turn.
    model = OpenAIModel(
        client_args={
            "api_key": config.llm_api_key,
            "base_url": config.llm_api_endpoint,
        },
        model_id=config.llm_model,
        params={"max_tokens": config.llm_max_tokens},
    )
    compaction_manager = ReMeCompactionManager(
        memory_provider=memory_provider,
        window_size=config.max_conversation_messages,
        compact_batch_size=config.compact_batch_size,
    )
    agent = Agent(
        model=model,
        tools=agent_tools,
        system_prompt=system_prompt,
        conversation_manager=compaction_manager,
        callback_handler=None,
        hooks=[ToolBudgetHookProvider()],
    )

    # Attach the per-loop state context so ``enforce_tool_budget`` (registered
    # via ``ToolBudgetHookProvider``) has budgets to consult on every tool
    # call.  The Orchestrator resets counters at the start of each turn.
    agent.state_context = AgentStateContext(
        max_tool_calls=config.max_tool_calls,
        per_tool_limits={
            "web_search": getattr(
                config, "tool_budget_web_search", DEFAULT_TOOL_BUDGET_WEB_SEARCH
            ),
            "run_python": getattr(
                config, "tool_budget_run_python", DEFAULT_TOOL_BUDGET_RUN_PYTHON
            ),
            "delegate_to_research": getattr(
                config,
                "tool_budget_delegate_to_research",
                DEFAULT_TOOL_BUDGET_DELEGATE_TO_RESEARCH,
            ),
        },
    )

    # --- Hooks ---
    post_session_hook = PostSessionHook(
        session_id=session_id,
        metrics_path=os.path.join(session_log_dir, DEFAULT_METRICS_FILENAME),
    )

    # --- Run ---
    orchestrator = Orchestrator(
        agent=agent,
        memory_provider=memory_provider,
        post_session_hook=post_session_hook,
        config=config,
        session_id=session_id,
        console=console,
    )

    try:
        orchestrator.run()
    finally:
        memory_provider.close()
        logger.info("main: session ended", extra={"data": {"session_id": session_id}})


if __name__ == "__main__":
    main()
