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

from src.config import Config
from src.logging_config import get_logger, init_session_logging
from src.memory.remelight.compaction_manager import ReMeCompactionManager
from src.memory.remelight.provider import ReMeLightProvider
from src.hooks.pre_session import PreSessionHook
from src.hooks.post_session import PostSessionHook
from src.prompts.loader import render_prompt
from src.tools.memorize import create_memorize_tool
from src.constants import (
    DEFAULT_METRICS_FILENAME,
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

    # Workaround for strands-agents issue #815
    import strands.tools.tools as _strands_tools
    import strands.event_loop.streaming as _strands_streaming
    from strands.tools.tools import InvalidToolUseNameException
    from src.orchestrator import tool_call_counter

    _patch_logger = get_logger(__name__)

    def _safe_validate_tool_use_name(tool: dict) -> None:
        """Guarded replacement for ``validate_tool_use_name``."""
        count = getattr(tool_call_counter, "count", 0) + 1
        limit = getattr(tool_call_counter, "limit", 0)
        tool_call_counter.count = count
        if limit and count > limit:
            _patch_logger.warning(
                "strands: tool call limit exceeded",
                extra={"data": {"count": count, "limit": limit}},
            )
            raise InvalidToolUseNameException(
                f"tool call limit reached ({count}/{limit}); stopping agent loop"
            )

        if not tool.get("name"):
            _patch_logger.warning(
                "strands #815: tool use with None/empty name intercepted",
                extra={"data": {"tool_use_id": tool.get("toolUseId"), "tool": str(tool)}},
            )
            raise InvalidToolUseNameException("tool name is None or empty (strands issue #815)")
        _orig_validate_tool_use_name(tool)

    _orig_validate_tool_use_name = _strands_tools.validate_tool_use_name
    _strands_tools.validate_tool_use_name = _safe_validate_tool_use_name
    _strands_streaming.validate_tool_use_name = _safe_validate_tool_use_name

    memorize_tool = create_memorize_tool(memory_provider, session_id=session_id)
    system_prompt = render_prompt("system_prompt")

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
        tools=[memorize_tool],
        system_prompt=system_prompt,
        conversation_manager=compaction_manager,
        callback_handler=None,
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
