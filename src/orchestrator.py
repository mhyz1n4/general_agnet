"""
Chat session orchestrator.

Per-turn flow:
  pre_mem_fetch hook → get_context_with_keys → post_mem_fetch hook
  → render context_prefix.j2 → agent(prefix + user_input)
  → save turn to memory → update metrics
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown

from src.logging_config import get_logger, set_trace_id
from src.prompts.loader import render_prompt

logger = get_logger(__name__)

EXIT_COMMANDS = {"exit", "quit", "bye", "/exit", "/quit"}


@dataclass
class SessionMetrics:
    start_time: float = field(default_factory=time.time)
    turn_count: int = 0
    memory_hits: int = 0
    memory_misses: int = 0
    turns: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": round(time.time() - self.start_time, 2),
            "turn_count": self.turn_count,
            "memory_hits": self.memory_hits,
            "memory_misses": self.memory_misses,
            "turns": self.turns,
        }


class Orchestrator:
    """Manages a single chat session."""

    def __init__(
        self,
        agent: Any,
        memory_manager: Any,
        pre_mem_fetch_hook: Any,
        post_mem_fetch_hook: Any,
        post_session_hook: Any,
        config: Any,
        session_id: str,
        console: Optional[Console] = None,
    ) -> None:
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

    def run(self) -> None:
        """Main chat loop."""
        self.console.print(
            "[bold green]Assistant ready.[/bold green] Type [bold]exit[/bold] to quit.\n"
        )

        while True:
            # Inactivity timeout check
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

            # New trace ID per turn
            trace_id = uuid.uuid4().hex
            set_trace_id(trace_id)

            try:
                response = self._process_turn(user_input)
            except Exception as exc:
                logger.error(
                    "orchestrator: turn failed",
                    extra={"data": {"error": str(exc)}},
                )
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
            normalised_query, limit=3
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

        # Step 4: render per-turn prefix and call agent
        prefix = render_prompt("context_prefix", memory_context=context)
        full_input = f"{prefix}{user_input}" if prefix.strip() else user_input

        logger.debug(
            "orchestrator: calling agent",
            extra={"data": {"query_len": len(full_input), "memory_keys": retrieved_keys}},
        )

        response = self.agent(full_input)
        response_text = str(response)

        # Step 5: save turn to memory
        turn_id = f"turn_{self.session_id}_{self.metrics.turn_count}"
        try:
            self.memory_manager.save_message(
                turn_id,
                f"User: {user_input}\nAssistant: {response_text}",
                {
                    "type": "episodic",
                    "session_id": self.session_id,
                    "turn": self.metrics.turn_count,
                },
            )
        except Exception as exc:
            logger.warning(
                "orchestrator: failed to save turn to memory",
                extra={"data": {"turn_id": turn_id, "error": str(exc)}},
            )

        # Step 6: update metrics
        self.metrics.turn_count += 1
        self.metrics.turns.append({"user": user_input, "assistant": response_text})

        logger.debug(
            "orchestrator: turn complete",
            extra={"data": {"turn": self.metrics.turn_count}},
        )
        return response_text

    def _close_session(self) -> None:
        """Trigger post-session hook (runs in daemon thread)."""
        logger.debug("orchestrator: closing session")
        try:
            self.post_session_hook.run(metrics=self.metrics.to_dict())
        except Exception as exc:
            logger.error(
                "orchestrator: post-session hook error",
                extra={"data": {"error": str(exc)}},
            )
