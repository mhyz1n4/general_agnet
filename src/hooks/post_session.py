"""
Post-session hook — runs in a daemon thread after the session closes.

Steps:
  1. Flush remaining Redis session messages to long-term filesystem memory.
  2. Call LLM to produce a one-paragraph session summary; save to conversations/.
  3. Write session metrics to logs/metrics.jsonl.

Failures are logged but never surfaced to the user.
"""

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

import redis as redis_lib

from src.client.base import LLMClient
from src.constants import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_METRICS_PATH,
    REDIS_DEFAULT_PREFIX,
    SUMMARY_MAX_TOKENS,
)
from src.logging_config import get_logger
from src.memory.manager import MemoryManager
from src.memory.types import SessionMetricsDict, TurnRecord
from .base import BaseHook, HookResult

logger = get_logger(__name__)


class PostSessionHook(BaseHook):
    """Flush session, summarise, and write metrics — in a daemon thread."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        session_id: str,
        redis_client: Optional[redis_lib.Redis] = None,
        redis_prefix: str = REDIS_DEFAULT_PREFIX,
        llm_client: Optional[LLMClient] = None,
        llm_model: str = DEFAULT_AGENT_MODEL,
        metrics_path: str = DEFAULT_METRICS_PATH,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        self.memory_manager = memory_manager
        self.session_id = session_id
        self.redis_client = redis_client
        self.redis_prefix = redis_prefix
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.metrics_path = metrics_path
        self.max_context_chars = max_context_chars

    def run(self, metrics: Optional[SessionMetricsDict] = None, **kwargs: object) -> HookResult:
        """Spawn a daemon thread to run post-session tasks."""
        metrics = metrics or {}
        thread = threading.Thread(
            target=self._run_async,
            args=(metrics,),
            daemon=True,
            name="post-session-hook",
        )
        thread.start()
        return HookResult(success=True, message="Post-session hook started in background.")

    # ------------------------------------------------------------------
    # Internal async work
    # ------------------------------------------------------------------

    def _run_async(self, metrics: SessionMetricsDict) -> None:
        """
        Execute all post-session tasks in sequence on the background thread.

        Steps: (1) flush Redis → filesystem, (2) generate LLM summary,
        (3) write metrics.  All steps catch and log their own exceptions so
        that a failure in one step does not prevent subsequent steps from
        running.

        Args:
            metrics: Session metrics dict from ``SessionMetrics.to_dict()``.
        """
        logger.debug("post_session: starting async tasks", extra={"data": {"session_id": self.session_id}})

        turns: List[TurnRecord] = metrics.get("turns", [])

        # Step 1: flush Redis → filesystem
        self._flush_redis()

        # Step 2: LLM summary
        self._write_summary(turns)

        # Step 3: metrics
        self._write_metrics(metrics)

        logger.debug("post_session: done", extra={"data": {"session_id": self.session_id}})

    def _flush_redis(self) -> None:
        """
        Copy remaining Redis session messages for this session to long-term filesystem memory.

        Scans all keys under the configured prefix, deserialises each record,
        and flushes only those whose ``metadata.session_id`` matches the current
        session.  This prevents the hook from flushing — and deleting — data that
        belongs to other concurrent sessions.

        Keys that fail individually are logged at ERROR and skipped; the loop
        continues.  No-ops if no ``redis_client`` is configured.
        """
        if self.redis_client is None:
            return
        try:
            all_redis_keys: List[str] = list(self.redis_client.scan_iter(f"{self.redis_prefix}*"))
            logger.debug(
                "post_session: scanning Redis keys",
                extra={"data": {"total_scanned": len(all_redis_keys), "session_id": self.session_id}},
            )
            flushed: int = 0
            for key in all_redis_keys:
                try:
                    raw = self.redis_client.get(key)
                    if raw is None:
                        continue
                    data = json.loads(raw)
                    meta: dict = data.get("metadata", {})

                    # Only flush entries that belong to this session.
                    if meta.get("session_id") != self.session_id:
                        continue

                    msg_id: str = data.get("id", key.decode() if isinstance(key, bytes) else str(key))
                    content: str = data.get("content", "")
                    self.memory_manager.save_message(msg_id, content, meta)
                    flushed += 1
                    logger.debug("post_session: flushed key", extra={"data": {"key": str(key)}})
                except Exception as exc:
                    logger.error(
                        "post_session: failed to flush key",
                        extra={"data": {"key": str(key), "error": str(exc)}},
                    )
            logger.debug("post_session: Redis flush complete", extra={"data": {"flushed": flushed}})
        except Exception as exc:
            logger.error(
                "post_session: Redis flush error",
                extra={"data": {"error": str(exc)}},
            )

    def _select_turns_for_summary(self, turns: List[TurnRecord]) -> List[TurnRecord]:
        """
        Select the most recent turns that fit within ``max_context_chars``.

        Iterates backwards through *turns*, accumulating the formatted character
        count for each turn until adding the next would exceed the budget.  This
        ties the turn limit to the same character budget used throughout the
        system rather than an arbitrary constant.

        Args:
            turns: Full list of turn records for the session (oldest first).

        Returns:
            A sublist of *turns* (oldest-first order preserved) whose total
            formatted length is at most ``max_context_chars``.
        """
        selected: List[TurnRecord] = []
        budget: int = self.max_context_chars
        for turn in reversed(turns):
            formatted: str = f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
            if len(formatted) > budget:
                # Skip this oversized turn but keep scanning older ones — a
                # large recent turn should not block smaller older turns from
                # being included in the summary.
                continue
            selected.append(turn)
            budget -= len(formatted)
        selected.reverse()
        logger.debug(
            "post_session: turns selected for summary",
            extra={"data": {"total": len(turns), "selected": len(selected)}},
        )
        return selected

    def _write_summary(self, turns: List[TurnRecord]) -> None:
        """
        Generate a one-paragraph LLM summary of the session and persist it.

        Selects the most recent turns that fit within ``max_context_chars`` (via
        ``_select_turns_for_summary``), formats them as alternating User/Assistant
        lines, and saves the resulting summary as an episodic memory entry via
        ``memory_manager.save_message()``.
        No-ops if no ``llm_client`` is configured or the turn list is empty.

        Args:
            turns: List of turn records for the completed session.
        """
        if self.llm_client is None or not turns:
            return
        try:
            recent = self._select_turns_for_summary(turns)
            turn_text = "\n".join(
                f"User: {t.get('user', '')}\nAssistant: {t.get('assistant', '')}"
                for t in recent
            )
            prompt = (
                f"Write a one-paragraph summary of this conversation session:\n\n{turn_text}"
            )
            response = self.llm_client.completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.llm_model,
                max_tokens=SUMMARY_MAX_TOKENS,
            )
            summary_text = ""
            if hasattr(response, "content") and response.content:
                block = response.content[0]
                summary_text = getattr(block, "text", str(block))

            summary_id = f"session_{self.session_id}"
            self.memory_manager.save_message(
                summary_id,
                summary_text,
                {
                    "type": "episodic",
                    "session_id": self.session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.debug("post_session: summary saved", extra={"data": {"id": summary_id}})
        except Exception as exc:
            logger.error(
                "post_session: summary generation failed",
                extra={"data": {"error": str(exc)}},
            )

    def _write_metrics(self, metrics: SessionMetricsDict) -> None:
        """
        Append a session metrics record to the JSONL metrics log.

        The ``turns`` list is omitted from the persisted record to keep the
        log compact.  The metrics directory is created if it does not exist.

        Args:
            metrics: Session metrics dict from ``SessionMetrics.to_dict()``.
        """
        try:
            os.makedirs(os.path.dirname(self.metrics_path) or ".", exist_ok=True)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": self.session_id,
                **{k: v for k, v in metrics.items() if k != "turns"},
            }
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            logger.debug("post_session: metrics written")
        except Exception as exc:
            logger.error(
                "post_session: metrics write failed",
                extra={"data": {"error": str(exc)}},
            )
