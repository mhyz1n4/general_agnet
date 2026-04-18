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
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from rich.console import Console
from rich.markdown import Markdown

from src.config import Config
from src.constants import (
    CHARS_PER_TOKEN,
    CONSECUTIVE_FAILURES_BEFORE_GRACEFUL,
    CONTEXT_BUDGET_ALERT_THRESHOLD,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_RETRIEVAL_LIMIT,
    EXIT_COMMANDS,
    MEMORY_TIMESTAMP_DISPLAY_FORMAT,
)
from src.hooks.post_session import PostSessionHook
from src.logging_config import get_logger, set_trace_id
from src.memory.provider import MemoryItem, MemoryProvider, SearchFilters
from src.memory.types import SessionMetricsDict, TurnRecord
from src.prompts.loader import render_prompt

logger = get_logger(__name__)

_GRACEFUL_ERROR = (
    "I'm having trouble processing your request right now. "
    "Please try again or rephrase your message."
)

# Thread-local state used by the Strands #815 workaround patch in main.py.
tool_call_counter = threading.local()  # attrs: count (int), limit (int)

_THINKING_TAG_RE = re.compile(r"<think(?:ing)?>\s*.*?\s*</think(?:ing)?>", re.DOTALL | re.IGNORECASE)

_QUERY_NORMALIZE_RE = re.compile(r"[^a-zA-Z0-9\s']")
_WHITESPACE_COLLAPSE_RE = re.compile(r"\s+")


def _strip_thinking(text: str) -> str:
    """Remove CoT thinking blocks from *text* and return the cleaned string."""
    return _THINKING_TAG_RE.sub("", text).strip()


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
                else:
                    self.console.print(f"[red]Error:[/red] {exc}")
                continue

            self.console.print("\n[bold cyan]Assistant:[/bold cyan]")
            self.console.print(Markdown(response))
            self.console.print()

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

        response_text = self._call_agent_with_timeout(full_input)

        self.metrics.llm_token_count_est += (len(full_input) + len(response_text)) // CHARS_PER_TOKEN

        try:
            ts_display = time.strftime(MEMORY_TIMESTAMP_DISPLAY_FORMAT, time.gmtime())
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

        self.metrics.turn_count += 1
        self.metrics.turns.append({"user": user_input, "assistant": response_text})

        logger.debug(
            "orchestrator: turn complete",
            extra={"data": {"turn": self.metrics.turn_count}},
        )
        return response_text

    def _call_agent_with_timeout(self, full_input: str) -> str:
        """
        Invoke the Strands agent with a wall-clock timeout.

        Raises TimeoutError if the call exceeds TOOL_TIMEOUT_SECONDS.
        """
        logger.debug(
            "orchestrator: prompt input to LLM",
            extra={"data": {"input_len": len(full_input), "input": full_input}},
        )

        timeout = self.config.tool_timeout_seconds
        max_calls = self.config.max_tool_calls
        t0 = time.perf_counter()

        ctx: contextvars.Context = contextvars.copy_context()

        def _invoke() -> object:
            tool_call_counter.count = 0
            tool_call_counter.limit = max_calls
            try:
                return ctx.run(self.agent, full_input)
            finally:
                self.metrics.tool_calls_made += getattr(tool_call_counter, "count", 0)

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

        raw_response = str(response)
        logger.debug(
            "orchestrator: raw LLM response",
            extra={"data": {"response_len": len(raw_response), "response": raw_response}},
        )
        return _strip_thinking(raw_response)

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
        metrics_dict = self.metrics.to_dict()
        metrics_dict["storage_size_bytes"] = self._dir_size(self.config.memory_root)
        try:
            self.post_session_hook.run(metrics=metrics_dict)
        except Exception as exc:
            logger.error(
                "orchestrator: post-session hook error",
                extra={"data": {"error": str(exc)}},
            )

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
