"""
Chat session orchestrator.

Per-turn flow:
  pre_mem_fetch hook → get_context_with_keys → post_mem_fetch hook
  → render context_prefix.j2 → agent(prefix + user_input)
  → save turn to memory → update metrics

Design features:
  - Tool timeout: every agent call is wrapped with TOOL_TIMEOUT_SECONDS.
  - Error recovery (§5c): tracks consecutive turn failures; after two in a row
    returns a graceful degraded message and resets the counter.
  - Context budget (§5b): logs a critical alert when injected context approaches
    the configured character limit.  Strands manages its own history so we cannot
    truncate it directly, but the alert signals operator intervention.
  - Enhanced metrics: duration, token estimate, latency, turn count, hit/miss,
    eviction count, index/storage sizes.
"""

import concurrent.futures
import contextvars
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

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
    MEMORY_TYPE_EPISODIC,
)
from src.hooks.post_mem_fetch import PostMemFetchHook
from src.hooks.post_session import PostSessionHook
from src.hooks.pre_mem_fetch import PreMemFetchHook
from src.logging_config import get_logger, set_trace_id
from src.memory.manager import MemoryManager
from src.memory.types import SessionMetricsDict, TurnRecord
from src.prompts.loader import render_prompt

logger = get_logger(__name__)

_GRACEFUL_ERROR = (
    "I'm having trouble processing your request right now. "
    "Please try again or rephrase your message."
)

# Thread-local state used by the Strands #815 workaround patch in main.py.
# Each agent invocation (which runs in its own executor thread) initialises
# these before calling the agent so the patch can enforce a per-invocation
# tool-call retry limit without shared mutable global state.
tool_call_counter = threading.local()  # attrs: count (int), limit (int)

# Matches <think>…</think> and <thinking>…</thinking> emitted by CoT/reasoning
# models (e.g. DeepSeek-R1, QwQ, o1-style open-weights).  The block is always
# stripped before the response reaches the user or memory — we never want raw
# chain-of-thought in either place.
_THINKING_TAG_RE = re.compile(r"<think(?:ing)?>\s*.*?\s*</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove CoT thinking blocks from *text* and return the cleaned string."""
    return _THINKING_TAG_RE.sub("", text).strip()


@dataclass
class SessionMetrics:
    """Accumulated metrics for one chat session."""

    start_time: float = field(default_factory=time.time)
    turn_count: int = 0
    memory_hits: int = 0
    memory_misses: int = 0
    llm_token_count_est: int = 0    # estimated: (input_chars + output_chars) // CHARS_PER_TOKEN
    llm_latency_ms: float = 0.0     # wall-clock ms spent inside agent() calls
    tool_calls_made: int = 0        # tracked externally where Strands exposes it
    tool_failures: int = 0          # turns where agent() raised an exception
    eviction_count: int = 0         # Redis→FS evictions (from MemoryManager)
    turns: List[TurnRecord] = field(default_factory=list)

    def to_dict(self) -> SessionMetricsDict:
        """
        Serialise the accumulated metrics to a plain dict for the post-session hook.

        Returns:
            A ``SessionMetricsDict`` with all tracked fields populated.
            ``duration_seconds`` is computed relative to ``start_time`` at
            the moment of this call.
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
            "eviction_count": self.eviction_count,
            "turns": self.turns,
        }


class Orchestrator:
    """Manages a single chat session."""

    def __init__(
        self,
        agent: Callable[[str], object],
        memory_manager: MemoryManager,
        pre_mem_fetch_hook: PreMemFetchHook,
        post_mem_fetch_hook: PostMemFetchHook,
        post_session_hook: PostSessionHook,
        config: Config,
        session_id: str,
        console: Optional[Console] = None,
    ) -> None:
        """
        Initialise the orchestrator for a single chat session.

        Args:
            agent:               Strands Agent callable.  Accepts a single string
                                 (the full prompt including injected context) and
                                 returns the model response as a string or
                                 string-coercible object.
            memory_manager:      Manages long-term memory reads and writes.
                                 ``get_context_with_keys`` is called on every turn;
                                 ``save_message`` stores the completed turn.
            pre_mem_fetch_hook:  Runs before ``get_context_with_keys`` to normalise
                                 the query (e.g. lowercase, strip punctuation).
            post_mem_fetch_hook: Runs after ``get_context_with_keys`` to truncate
                                 context exceeding ``config.max_context_chars`` and
                                 optionally log the retrieval event.
            post_session_hook:   Fired once when the chat loop exits.  Writes
                                 metrics, generates a session summary, and flushes
                                 the Redis session store to long-term storage.
            config:              Validated application settings (timeouts, limits,
                                 model ID, etc.).
            session_id:          Unique identifier for this chat session (UUID hex).
                                 Used in log correlation, file paths, and Redis key
                                 scoping.
            console:             Rich Console instance for rendering output.
                                 Defaults to a fresh ``Console()`` if not provided.
        """
        self.agent = agent
        self.memory_manager = memory_manager
        self.pre_mem_fetch_hook = pre_mem_fetch_hook
        self.post_mem_fetch_hook = post_mem_fetch_hook
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
        # Step 1: pre-mem-fetch — normalise query
        pre_result = self.pre_mem_fetch_hook.run(user_input)
        normalised_query = pre_result.message if pre_result.success else user_input

        # Step 2: retrieve memory context
        context, retrieved_keys = self.memory_manager.get_context_with_keys(
            normalised_query, limit=DEFAULT_RETRIEVAL_LIMIT
        )

        if context:
            self.metrics.memory_hits += 1
        else:
            self.metrics.memory_misses += 1

        # Step 3: post-mem-fetch — truncate if needed
        post_result = self.post_mem_fetch_hook.run(
            context=context,
            retrieved_keys=retrieved_keys,
            query=normalised_query,
        )
        context = post_result.message if post_result.success else context

        # Step 4: context budget check (§5b)
        prefix = render_prompt("context_prefix", memory_context=context)
        full_input = f"{prefix}{user_input}" if prefix.strip() else user_input
        self._check_context_budget(full_input)

        # Step 5: call agent with timeout
        response_text = self._call_agent_with_timeout(full_input)

        # Step 6: update token/latency metrics (estimated)
        self.metrics.llm_token_count_est += (len(full_input) + len(response_text)) // CHARS_PER_TOKEN

        # Step 7: save turn to memory with timestamp embedded in content so the
        # LLM can answer temporal questions ("what did I say on Tuesday?").
        turn_id = f"turn_{self.session_id}_{self.metrics.turn_count}"
        try:
            ts_str = time.strftime(MEMORY_TIMESTAMP_DISPLAY_FORMAT, time.gmtime())
            self.memory_manager.save_message(
                turn_id,
                f"[{ts_str}]\nUser: {user_input}\nAssistant: {response_text}",
                {
                    "type": MEMORY_TYPE_EPISODIC,
                    "session_id": self.session_id,
                    "turn": self.metrics.turn_count,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        except Exception as exc:
            logger.warning(
                "orchestrator: failed to save turn to memory",
                extra={"data": {"turn_id": turn_id, "error": str(exc)}},
            )

        # Step 8: update metrics
        self.metrics.turn_count += 1
        self.metrics.turns.append({"user": user_input, "assistant": response_text})
        self.metrics.eviction_count = self.memory_manager.eviction_count

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

        # Capture the current ContextVar state (including trace_id) so that the
        # executor thread inherits it.  Without this, ContextVar values set on
        # the main thread (e.g. set_trace_id) are invisible to the worker thread
        # and all tool-call logs would show trace_id="unset".
        ctx: contextvars.Context = contextvars.copy_context()

        def _invoke() -> object:
            # threading.local is per-thread; reset here so each agent invocation
            # starts with a fresh counter regardless of thread reuse.
            tool_call_counter.count = 0
            tool_call_counter.limit = max_calls
            return ctx.run(self.agent, full_input)

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
        configured character limit (§5b).

        Note: Strands manages conversation history internally so we cannot
        truncate it here.  This alert signals that the operator should either
        increase max_context_chars or that summarisation is needed.

        ``max_context_chars`` is already a character limit — do NOT multiply
        it by CHARS_PER_TOKEN here; that would inflate the threshold 4×.
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
        """Collect final storage metrics then trigger post-session hook."""
        logger.debug("orchestrator: closing session")
        metrics_dict = self.metrics.to_dict()
        metrics_dict["index_size_bytes"] = self._dir_size(
            os.path.dirname(self.config.index_path) or "."
        )
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
