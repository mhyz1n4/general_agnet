"""
Entry point for the personal assistant agent.

Wire all components, run the PreSessionHook, then start the Orchestrator chat loop.
"""

import os
import sys
import uuid

from rich.console import Console

# Load config early (fails fast on missing required env vars)
from src.config import Config
from src.logging_config import get_logger
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
from src.orchestrator import Orchestrator

console = Console()


def _try_connect_redis(config: Config):
    """Attempt to connect to Redis; return client or None on failure."""
    try:
        import redis

        client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True,
        )
        client.ping()
        return client
    except Exception:
        return None


def main() -> None:
    config = Config()
    logger = get_logger(__name__, log_path=config.log_path, log_level=config.log_level)

    session_id = uuid.uuid4().hex[:12]
    logger.info("main: starting session", extra={"data": {"session_id": session_id}})

    # --- Memory components ---
    storage = TypedMarkdownStorage(memory_root=config.memory_root)
    indexer = JSONIndexer(index_path=config.index_path)
    retriever = KeywordRetriever(indexer=indexer)
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
    )

    # --- Redis (optional) ---
    redis_client = _try_connect_redis(config)
    if redis_client:
        from src.memory.redis.storage import RedisStorage

        session_storage = RedisStorage(
            host=config.redis_host,
            port=config.redis_port,
            default_ttl=config.redis_ttl,
        )
        memory_manager.session_storage = session_storage
        logger.info("main: Redis connected")
    else:
        redis_client = None
        logger.warning("main: Redis unavailable, continuing without session storage")

    # --- Pre-session hook ---
    from src.client.claude import ClaudeClient

    llm_client = ClaudeClient(api_key=config.llm_api_key)
    pre_session_hook = PreSessionHook(
        memory_root=config.memory_root,
        redis_client=redis_client,
        llm_client=llm_client,
        log_dir=os.path.dirname(config.log_path) or "./logs",
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
        from strands.models.anthropic import AnthropicModel
    except ImportError:
        console.print("[red]strands-agents package not installed. Run: pip install strands-agents[anthropic][/red]")
        sys.exit(1)

    memorize_tool = create_memorize_tool(memory_manager)
    system_prompt = render_prompt("system_prompt")

    model = AnthropicModel(
        model_id=config.llm_model,
        max_tokens=config.llm_max_tokens,
    )
    agent = Agent(
        model=model,
        tools=[memorize_tool],
        system_prompt=system_prompt,
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
        metrics_path=os.path.join(os.path.dirname(config.log_path) or "./logs", "metrics.jsonl"),
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
