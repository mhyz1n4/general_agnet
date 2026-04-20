"""
Chat session orchestrator.

Per-turn flow (V1.1):
  inline normalize query -> memory_provider.search() -> render context
  (trimming low-relevance items to fit max_context_chars)
  -> agent(prefix + user_input) -> memory_provider.save() -> update metrics

Design features:
  - Tool timeout: every agent call is wrapped with TOOL_TIMEOUT_SECONDS.
  - Error recovery (SS5c): tracks consecutive turn failures; after two in a row
    returns a graceful degraded message and resets the counter.
  - Context budget (SS5b): logs a critical alert when injected context approaches
    the configured character limit.  Strands manages its own history so we cannot
    truncate it directly, but the alert signals operator intervention.
  - Enhanced metrics: duration, token estimate, latency, turn count, hit/miss.
"""

import concurrent.futures
import contextvars
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from rich.console import Console
from rich.markdown import Markdown

from src.agents.context import AgentStateContext
from src.config import Config
from src.constants import (
    CHARS_PER_TOKEN,
    CONSECUTIVE_FAILURES_BEFORE_GRACEFUL,
    CONTEXT_BUDGET_ALERT_THRESHOLD,
    DEFAULT_RETRIEVAL_LIMIT,
    DEFAULT_TOOL_BUDGET_DELEGATE_TO_RESEARCH,
    DEFAULT_TOOL_BUDGET_RUN_PYTHON,
    DEFAULT_TOOL_BUDGET_WEB_SEARCH,
    EXIT_COMMANDS,
    MEMORY_TIMESTAMP_DISPLAY_FORMAT,
    METRICS_FLUSH_INTERVAL_TURNS,
)
from src.hooks.post_session import PostSessionHook
from src.logging_config import get_logger, get_trace_id, set_trace_id
from src.memory.provider import MemoryItem, MemoryProvider, SearchFilters
from src.memory.types import SessionMetricsDict, TurnRecord
from src.prompts.loader import render_prompt
from src.tools.cache import SessionToolCache, set_session_cache

logger = get_logger(__name__)

_GRACEFUL_ERROR = (
    "I'm having trouble processing your request right now. "
    "Please try again or rephrase your message."
)


_THINKING_TAG_RE = re.compile(r"<think(?:ing)?>\s*(.*?)\s*</think(?:ing)?>", re.DOTALL | re.IGNORECASE)

_QUERY_NORMALIZE_RE = re.compile(r"[^a-zA-Z0-9\s']")
_WHITESPACE_COLLAPSE_RE = re.compile(r"\s+")


def _split_thinking(text: str) -> tuple[str, str]:
    """
    Separate chain-of-thought blocks from the user-visible response.

    The model may emit one or more ``<think>...</think>`` (or ``<thinking>``)
    spans; we surface them in logs and on the console for transparency, but
    strip them from whatever is persisted to memory so the CoT trace does
    not pollute long-term storage.

    Returns:
        ``(thinking, visible)`` — ``thinking`` concatenates all CoT spans
        separated by blank lines (empty string if none); ``visible`` is the
        cleaned response with CoT spans removed.
    """
    thoughts = [m.strip() for m in _THINKING_TAG_RE.findall(text) if m.strip()]
    visible = _THINKING_TAG_RE.sub("", text).strip()
    return "\n\n".join(thoughts), visible


def _strip_thinking(text: str) -> str:
    """Backwards-compatible helper that returns only the visible portion."""
    return _split_thinking(text)[1]


def _normalize_query(query: str) -> str:
    """Normalize a user query for memory search: strip punctuation, lowercase, collapse whitespace."""
    cleaned = _QUERY_NORMALIZE_RE.sub("", query)
    cleaned = cleaned.lower()
    return _WHITESPACE_COLLAPSE_RE.sub(" ", cleaned).strip()


def _build_context_input(
    results: List[MemoryItem],
    user_input: str,
    max_chars: int,
) -> tuple[str, int]:
    """Render context_prefix + user_input, dropping tail items that exceed budget.

    Items are added in order (assumed most-relevant-first) and dropped from the
    tail until the rendered ``prefix + user_input`` fits within ``max_chars``.

    Args:
        results:    Memory items ordered most-relevant-first.
        user_input: Raw user query (always included verbatim).
        max_chars:  Hard cap on the returned string length.

    Returns:
        ``(full_input, dropped_count)`` — the agent input and how many tail
        items were excluded by the budget.
    """
    def _render(items: List[MemoryItem]) -> str:
        context = "\n\n".join(f"[{r.type}] {r.content}" for r in items) if items else ""
        prefix = render_prompt("context_prefix", memory_context=context)
        return f"{prefix}{user_input}" if prefix.strip() else user_input

    kept: List[MemoryItem] = []
    full_input: Optional[str] = None
    for item in results:
        candidate = _render(kept + [item])
        if len(candidate) > max_chars:
            break
        kept.append(item)
        full_input = candidate
    if full_input is None:
        full_input = _render(kept)
    return full_input, len(results) - len(kept)


@dataclass
class SessionMetrics:
    """Accumulated metrics for one chat session."""

    start_time: float = field(default_factory=time.time)
    turn_count: int = 0
    memory_hits: int = 0
    memory_misses: int = 0
    llm_token_count_est: int = 0
    llm_latency_ms: float = 0.0
    tool_calls_made: int = 0
    tool_failures: int = 0
    cache_hits: int = 0
    cache_read_input_tokens: int = 0
    turns: List[TurnRecord] = field(default_factory=list)

    def to_dict(self) -> SessionMetricsDict:
        """
        Serialise the accumulated metrics to a plain dict for the post-session hook.

        Returns:
            A ``SessionMetricsDict`` with all tracked fields populated.
        """
        return {
            "duration_seconds": round(time.time() - self.start_time, 2),
            "turn_count": self.turn_count,
            "memory_hits": self.memory_hits,
            "memory_misses": self.memory_misses,
            "llm_token_count_est": self.llm_token_count_est,
            "llm_latency_ms": round(self.llm_latency_ms, 1),
            "tool_calls_made": self.tool_calls_made,
            "tool_failures": self.tool_failures,
            "cache_hits": self.cache_hits,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "turns": self.turns,
        }


class Orchestrator:
    """Manages a single chat session."""

    def __init__(
        self,
        agent: Callable[[str], object],
        memory_provider: MemoryProvider,
        post_session_hook: PostSessionHook,
        config: Config,
        session_id: str,
        console: Optional[Console] = None,
    ) -> None:
        """
        Initialise the orchestrator for a single chat session.

        Args:
            agent:               Strands Agent callable.
            memory_provider:     ``MemoryProvider`` implementation for search,
                                 save, context management, and session history.
            post_session_hook:   Fired once when the chat loop exits.  Writes
                                 session metrics.
            config:              Validated application settings.
            session_id:          Unique identifier for this chat session.
            console:             Rich Console instance for rendering output.
        """
        self.agent = agent
        self.memory_provider = memory_provider
        self.post_session_hook = post_session_hook
        self.config = config
        self.session_id = session_id
        self.console = console or Console()
        self.metrics = SessionMetrics()
        self._last_input_time = time.time()
        self._consecutive_failures: int = 0
        self._tool_cache = SessionToolCache()

        # Attach an ``AgentStateContext`` if the caller didn't bring one.
        # ``enforce_tool_budget`` (wired via ``ToolBudgetHookProvider``) reads
        # this on every ``BeforeToolCallEvent``.
        if getattr(self.agent, "state_context", None) is None:
            self.agent.state_context = AgentStateContext(
                max_tool_calls=getattr(self.config, "max_tool_calls", 0) or 0,
                per_tool_limits=self._per_tool_budget(),
            )

    def run(self) -> None:
        """Main chat loop."""
        self.console.print(
            "[bold green]Assistant ready.[/bold green] Type [bold]exit[/bold] to quit.\n"
        )

        while True:
            idle = time.time() - self._last_input_time
            if idle > self.config.session_inactivity_timeout_seconds:
                self.console.print("\n[yellow]Session timed out due to inactivity.[/yellow]")
                break

            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[yellow]Session interrupted.[/yellow]")
                break

            if not user_input:
                continue

            self._last_input_time = time.time()

            if user_input.lower() in EXIT_COMMANDS:
                self.console.print("[yellow]Goodbye![/yellow]")
                break

            trace_id = uuid.uuid4().hex
            set_trace_id(trace_id)

            try:
                response = self._process_turn(user_input)
                self._consecutive_failures = 0
            except Exception as exc:
                self._consecutive_failures += 1
                self.metrics.tool_failures += 1
                logger.error(
                    "orchestrator: turn failed",
                    extra={"data": {
                        "error": str(exc),
                        "consecutive_failures": self._consecutive_failures,
                    }},
                )
                if self._consecutive_failures >= CONSECUTIVE_FAILURES_BEFORE_GRACEFUL:
                    logger.error(
                        "orchestrator: two consecutive turn failures — returning graceful error",
                        extra={"data": {"alert": True}},
                    )
                    self._consecutive_failures = 0
                    self.console.print(f"\n[red]Assistant:[/red] {_GRACEFUL_ERROR}\n")
                    # Snapshot metrics so the failure isn't lost if the user
                    # closes the terminal without typing ``exit``.
                    self._flush_metrics_snapshot()
                else:
                    self.console.print(f"[red]Error:[/red] {exc}")
                continue

            self.console.print("\n[bold cyan]Assistant:[/bold cyan]")
            self.console.print(Markdown(response))
            self.console.print()

            # Periodic snapshot — bounds metric loss to N turns when the
            # process is killed without a clean exit.
            if (
                self.metrics.turn_count > 0
                and self.metrics.turn_count % METRICS_FLUSH_INTERVAL_TURNS == 0
            ):
                self._flush_metrics_snapshot()

        self._close_session()

    def _process_turn(self, user_input: str) -> str:
        """Execute one full conversation turn and return the assistant response."""
        normalised_query = _normalize_query(user_input)

        results = self.memory_provider.search(
            normalised_query, SearchFilters(limit=DEFAULT_RETRIEVAL_LIMIT)
        )

        if results:
            self.metrics.memory_hits += 1
        else:
            self.metrics.memory_misses += 1

        full_input, dropped = _build_context_input(
            results, user_input, self.config.max_context_chars
        )
        if dropped:
            logger.warning(
                "orchestrator: dropped memory items to fit context budget",
                extra={"data": {"dropped": dropped, "kept": len(results) - dropped}},
            )

        # Alerts when user_input alone blows past the budget — trimming only covers memory items.
        self._check_context_budget(full_input)

        raw_response = self._call_agent_with_timeout(full_input)
        thinking, response_text = _split_thinking(raw_response)

        # Thinking goes to logs + console only — never to memory.  Logged at
        # INFO so it lands in ``agent.jsonl`` alongside tool traces.
        if thinking:
            logger.info(
                "orchestrator: model thinking",
                extra={"data": {"thinking_len": len(thinking), "thinking": thinking}},
            )

        self.metrics.llm_token_count_est += (len(full_input) + len(response_text)) // CHARS_PER_TOKEN

        ts_display = time.strftime(MEMORY_TIMESTAMP_DISPLAY_FORMAT, time.gmtime())
        try:
            self.memory_provider.save(
                f"[{ts_display}]\nUser: {user_input}\nAssistant: {response_text}",
                type="episodic",
                topic=self.session_id,
            )
        except Exception as exc:
            logger.warning(
                "orchestrator: failed to save episodic memory",
                extra={"data": {"error": str(exc)}},
            )

        # One episodic entry per tool invoked, tagged with the tool name so
        # future retrieval can filter by topic.  Tool names come from the
        # ``enforce_tool_budget`` hook, which appends every successful call
        # to ``state_context.tools_invoked``.
        tool_names = self.agent.state_context.unique_tools_invoked()
        for tool_name in tool_names:
            try:
                self.memory_provider.save(
                    f"[{ts_display}] tool={tool_name} summary: {response_text}",
                    type="episodic",
                    topic=tool_name,
                )
            except Exception as exc:
                logger.warning(
                    "orchestrator: tool-summary write-back failed",
                    extra={"data": {"tool": tool_name, "error": str(exc)}},
                )

        self.metrics.turn_count += 1
        self.metrics.turns.append({"user": user_input, "assistant": response_text})

        logger.debug(
            "orchestrator: turn complete",
            extra={"data": {"turn": self.metrics.turn_count}},
        )
        return thinking, response_text

    def _call_agent_with_timeout(self, full_input: str) -> str:
        """
        Invoke the Strands agent with a wall-clock timeout.

        Raises TimeoutError if the call exceeds TOOL_TIMEOUT_SECONDS.
        """
        logger.debug(
            "orchestrator: prompt input to LLM",
            extra={"data": {"input_len": len(full_input), "input": full_input}},
        )

        timeout = self.config.agent_turn_timeout_seconds
        t0 = time.perf_counter()

        ctx: contextvars.Context = contextvars.copy_context()
        # Bind the per-session cache on the copied context so @cached_tool
        # decorators can find it on the worker thread.
        ctx.run(set_session_cache, self._tool_cache)

        # Reset the agent's state context for this turn.  The hook populated
        # by ``ToolBudgetHookProvider`` will read it on every tool call.
        state = self.agent.state_context
        state.reset(trace_id=get_trace_id())

        def _invoke() -> object:
            """Run the agent inside the copied context and roll up metrics."""
            hits_before = self._tool_cache.hits
            try:
                return ctx.run(self.agent, full_input)
            finally:
                self.metrics.tool_calls_made += state.tool_call_count
                self.metrics.cache_hits += self._tool_cache.hits - hits_before

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            try:
                response = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Agent call timed out after {timeout}s"
                )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.metrics.llm_latency_ms += elapsed_ms

        # Strands ``accumulated_usage`` is cumulative across the session, so
        # we assign rather than add.  Backends without cache-control support
        # (e.g. local vLLM) leave the field absent/zero.
        usage = getattr(
            getattr(self.agent, "event_loop_metrics", None), "accumulated_usage", None
        )
        if isinstance(usage, dict):
            self.metrics.cache_read_input_tokens = int(usage.get("cacheReadInputTokens", 0) or 0)

        raw_response = str(response)
        logger.debug(
            "orchestrator: raw LLM response",
            extra={"data": {"response_len": len(raw_response), "response": raw_response}},
        )
        return raw_response

    def _check_context_budget(self, full_input: str) -> None:
        """
        Emit a critical alert when the injected turn input approaches the
        configured character limit.
        """
        budget = self.config.max_context_chars
        if len(full_input) > budget * CONTEXT_BUDGET_ALERT_THRESHOLD:
            logger.error(
                "orchestrator: context approaching token limit",
                extra={"data": {
                    "chars": len(full_input),
                    "approx_tokens": len(full_input) // CHARS_PER_TOKEN,
                    "budget_chars": budget,
                    "alert": True,
                }},
            )

    def _close_session(self) -> None:
        """Flush remaining dialog, collect storage metrics, run post-session hook."""
        logger.debug("orchestrator: closing session")
        conv_manager = getattr(self.agent, "conversation_manager", None)
        flush = getattr(conv_manager, "flush", None)
        if callable(flush):
            try:
                flush(self.agent)
            except Exception as exc:
                logger.warning(
                    "orchestrator: conversation_manager.flush failed",
                    extra={"data": {"error": str(exc)}},
                )
        self._flush_metrics_snapshot()

    def _flush_metrics_snapshot(self) -> None:
        """
        Append the current metrics dict to ``metrics.jsonl``.

        Called periodically from the main loop and on graceful failure so
        that an abrupt process exit (no ``exit`` command, terminal closed)
        loses at most ``METRICS_FLUSH_INTERVAL_TURNS`` turns of metrics.
        Each call appends one line — the file is a time-series; the last
        line is the most current snapshot.
        """
        metrics_dict = self.metrics.to_dict()
        metrics_dict["storage_size_bytes"] = self._dir_size(self.config.memory_root)
        try:
            self.post_session_hook.run(metrics=metrics_dict)
        except Exception as exc:
            logger.error(
                "orchestrator: post-session hook error",
                extra={"data": {"error": str(exc)}},
            )

    def _per_tool_budget(self) -> dict:
        """
        Return the per-tool call budget dict, pulled from the active Config.

        Missing config fields fall back to the module defaults so tests that
        use ``MagicMock(spec=Config)`` without setting every knob keep working.
        """
        return {
            "web_search": getattr(
                self.config, "tool_budget_web_search", DEFAULT_TOOL_BUDGET_WEB_SEARCH
            ),
            "run_python": getattr(
                self.config, "tool_budget_run_python", DEFAULT_TOOL_BUDGET_RUN_PYTHON
            ),
            "delegate_to_research": getattr(
                self.config,
                "tool_budget_delegate_to_research",
                DEFAULT_TOOL_BUDGET_DELEGATE_TO_RESEARCH,
            ),
        }

    @staticmethod
    def _dir_size(path: str) -> int:
        """Return total byte size of all files under path."""
        total = 0
        try:
            for dirpath, _, filenames in os.walk(path):
                for fname in filenames:
                    try:
                        total += os.path.getsize(os.path.join(dirpath, fname))
                    except OSError:
                        pass
        except OSError:
            pass
        return total
