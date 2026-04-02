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
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.logging_config import get_logger
from .base import BaseHook, HookResult

logger = get_logger(__name__)


class PostSessionHook(BaseHook):
    """Flush session, summarise, and write metrics — in a daemon thread."""

    def __init__(
        self,
        memory_manager: Any,
        session_id: str,
        redis_client: Optional[Any] = None,
        redis_prefix: str = "mem:",
        llm_client: Optional[Any] = None,
        llm_model: str = "claude-sonnet-4-6",
        metrics_path: str = "./logs/metrics.jsonl",
    ) -> None:
        self.memory_manager = memory_manager
        self.session_id = session_id
        self.redis_client = redis_client
        self.redis_prefix = redis_prefix
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.metrics_path = metrics_path

    def run(self, metrics: Optional[Dict[str, Any]] = None, **kwargs: Any) -> HookResult:
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

    def _run_async(self, metrics: Dict[str, Any]) -> None:
        logger.debug("post_session: starting async tasks", extra={"data": {"session_id": self.session_id}})

        turns: List[Dict] = metrics.get("turns", [])

        # Step 1: flush Redis → filesystem
        self._flush_redis()

        # Step 2: LLM summary
        self._write_summary(turns)

        # Step 3: metrics
        self._write_metrics(metrics)

        logger.debug("post_session: done", extra={"data": {"session_id": self.session_id}})

    def _flush_redis(self) -> None:
        if self.redis_client is None:
            return
        try:
            keys = list(self.redis_client.scan_iter(f"{self.redis_prefix}*"))
            logger.debug(
                "post_session: flushing Redis keys",
                extra={"data": {"count": len(keys)}},
            )
            for key in keys:
                try:
                    raw = self.redis_client.get(key)
                    if raw is None:
                        continue
                    data = json.loads(raw)
                    msg_id = data.get("id", key.decode() if isinstance(key, bytes) else key)
                    content = data.get("content", "")
                    meta = data.get("metadata", {})
                    self.memory_manager.save_message(msg_id, content, meta)
                    logger.debug("post_session: flushed key", extra={"data": {"key": str(key)}})
                except Exception as exc:
                    logger.error(
                        "post_session: failed to flush key",
                        extra={"data": {"key": str(key), "error": str(exc)}},
                    )
        except Exception as exc:
            logger.error(
                "post_session: Redis flush error",
                extra={"data": {"error": str(exc)}},
            )

    def _write_summary(self, turns: List[Dict]) -> None:
        if self.llm_client is None or not turns:
            return
        try:
            # Pass last 20 turns to avoid context overflow
            recent = turns[-20:]
            turn_text = "\n".join(
                f"User: {t.get('user', '')}\nAssistant: {t.get('assistant', '')}"
                for t in recent
            )
            prompt = (
                f"Write a one-paragraph summary of this conversation session:\n\n{turn_text}"
            )
            response = self.llm_client.completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
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

    def _write_metrics(self, metrics: Dict[str, Any]) -> None:
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
