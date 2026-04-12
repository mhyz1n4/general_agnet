"""
Entry point for the personal assistant agent.

Wire all components, run the PreSessionHook, then start the Orchestrator chat loop.
"""

import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console

# Load config early (fails fast on missing required env vars)
from src.config import Config
from src.logging_config import get_logger, init_session_logging
from src.memory.file_system.indexer import JSONIndexer
from src.memory.file_system.retriever import KeywordRetriever
from src.memory.file_system.typed_storage import TypedMarkdownStorage
from src.memory.manager import MemoryManager
from src.memory.dlq import DeadLetterQueue
from src.query.classifier import RegexClassifier
from src.query.preprocessor import TemporalExtractor
from src.hooks.pre_session import PreSessionHook
from src.hooks.post_session import PostSessionHook
from src.hooks.pre_mem_fetch import PreMemFetchHook
from src.hooks.post_mem_fetch import PostMemFetchHook
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


def _try_connect_redis(config: Config) -> Optional["redis.Redis"]:
    """
    Attempt to connect to Redis and verify with a PING.

    Returns:
        A connected ``redis.Redis`` client, or ``None`` if the connection fails
        for any reason.  Failures are logged at WARNING so operators know why
        session storage is unavailable.
    """
    try:
        import redis

        client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True,
        )
        client.ping()
        return client
    except Exception as exc:
        # Log the concrete error so operators can diagnose connection issues
        # (wrong host, auth failure, network partition, etc.) rather than
        # seeing a silent fallback with no diagnostic information.
        import logging
        logging.getLogger(__name__).warning(
            "main: Redis connection failed — continuing without session storage",
            extra={"data": {
                "host": config.redis_host,
                "port": config.redis_port,
                "error": str(exc),
            }},
        )
        return None


def main() -> None:
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

    # --- Memory components ---
    storage = TypedMarkdownStorage(memory_root=config.memory_root)
    indexer = JSONIndexer(index_path=config.index_path)
    retriever = KeywordRetriever(index_path=config.index_path, storage=storage)
    classifier = RegexClassifier()
    temporal_extractor = TemporalExtractor()
    dlq = DeadLetterQueue(dlq_path=config.dlq_path, max_attempts=config.dlq_max_attempts)

    memory_manager = MemoryManager(
        storage=storage,
        indexer=indexer,
        retriever=retriever,
        classifier=classifier,
        temporal_extractor=temporal_extractor,
        dlq=dlq,
        session_max_messages=config.session_max_messages,
        session_id=session_id,
    )

    # --- Redis (optional) ---
    redis_client = _try_connect_redis(config)
    if redis_client:
        from src.memory.redis.storage import RedisStorage

        session_storage = RedisStorage(
            client=redis_client,
            default_ttl=config.redis_ttl,
        )
        memory_manager.session_storage = session_storage
        logger.info("main: Redis connected")
    else:
        redis_client = None
        logger.warning("main: Redis unavailable, continuing without session storage")

    # --- LLM client (OpenAI-compatible; points to local vLLM by default) ---
    from src.client.open_ai_model import OpenAIModelClient

    llm_client = OpenAIModelClient(
        api_endpoint=config.llm_api_endpoint,
        token=config.llm_api_key,
    )

    # --- Pre-session hook ---
    pre_session_hook = PreSessionHook(
        memory_root=config.memory_root,
        redis_client=redis_client,
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
        # If Redis check failed, disable session_storage
        if "Redis" in pre_result.message:
            memory_manager.session_storage = None
            redis_client = None

    # --- Start DLQ background retry ---
    dlq_stop_event = dlq.start_background_retry(
        memory_manager, interval_seconds=config.dlq_retry_interval_seconds
    )

    # --- Strands Agent ---
    try:
        from strands import Agent
        from strands.models.openai import OpenAIModel
    except ImportError:
        console.print("[red]strands-agents package not installed. Run: pip install strands-agents[/red]")
        sys.exit(1)

    # Workaround for strands-agents issue #815:
    # Thinking models (e.g. DeepSeek-R1 via vLLM) stream tool calls where the
    # first chunk has function.name=None.  validate_tool_use_name checks for the
    # key's presence (passes) then calls re.match(pattern, None) → TypeError.
    #
    # Both callers already catch InvalidToolUseNameException and recover gracefully:
    #   - _validator.validate_and_prepare_tools: returns an error ToolResult to LLM
    #   - streaming._normalize_messages: replaces name with "INVALID_TOOL_NAME"
    # Neither catch TypeError, so the process crashes instead of recovering.
    #
    # Fix: patch validate_tool_use_name at every binding where it is used:
    #   1. strands.tools.tools  — source definition + called by validate_tool_use
    #   2. strands.event_loop.streaming — imported into its own module namespace
    import strands.tools.tools as _strands_tools
    import strands.event_loop.streaming as _strands_streaming
    from strands.tools.tools import InvalidToolUseNameException
    from src.orchestrator import tool_call_counter

    _patch_logger = get_logger(__name__)

    def _safe_validate_tool_use_name(tool: dict) -> None:
        # Enforce per-invocation tool-call retry limit (strands issue #815 / config.max_tool_calls).
        # tool_call_counter is threading.local so each agent invocation has its own counter;
        # the orchestrator resets count/limit before each agent() call.
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
        # Delegate to the original logic for all other cases
        _orig_validate_tool_use_name(tool)

    _orig_validate_tool_use_name = _strands_tools.validate_tool_use_name
    _strands_tools.validate_tool_use_name = _safe_validate_tool_use_name        # fixes tools.py globals
    _strands_streaming.validate_tool_use_name = _safe_validate_tool_use_name    # fixes streaming.py local binding

    memorize_tool = create_memorize_tool(memory_manager, session_id=session_id)
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
        # Suppress Strands' streaming printer — thinking blocks and the response
        # itself would both be printed raw to stdout before we can strip them.
        # The orchestrator displays the cleaned response via rich.Console.
        callback_handler=None,
    )

    # --- Hooks ---
    pre_mem_fetch_hook = PreMemFetchHook()
    post_mem_fetch_hook = PostMemFetchHook(
        max_context_chars=config.max_context_chars,
        session_storage=memory_manager.session_storage,
    )
    post_session_hook = PostSessionHook(
        memory_manager=memory_manager,
        session_id=session_id,
        redis_client=redis_client,
        llm_client=llm_client,
        llm_model=config.llm_model,
        metrics_path=os.path.join(session_log_dir, DEFAULT_METRICS_FILENAME),
        max_context_chars=config.max_context_chars,
    )

    # --- Run ---
    orchestrator = Orchestrator(
        agent=agent,
        memory_manager=memory_manager,
        pre_mem_fetch_hook=pre_mem_fetch_hook,
        post_mem_fetch_hook=post_mem_fetch_hook,
        post_session_hook=post_session_hook,
        config=config,
        session_id=session_id,
        console=console,
    )

    try:
        orchestrator.run()
    finally:
        dlq_stop_event.set()
        logger.info("main: session ended", extra={"data": {"session_id": session_id}})


if __name__ == "__main__":
    main()
