"""
Post-session hook — runs in a daemon thread after the session closes.

V1.1: only writes session metrics to logs/metrics.jsonl.
Redis flushing and LLM summary generation have been retired; the
MemoryProvider handles session persistence and compaction directly.
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Optional

from src.constants import DEFAULT_METRICS_PATH
from src.logging_config import get_logger
from src.memory.types import SessionMetricsDict
from .base import BaseHook, HookResult

logger = get_logger(__name__)


class PostSessionHook(BaseHook):
    """Write session metrics in a daemon thread after the session ends."""

    def __init__(
        self,
        session_id: str,
        metrics_path: str = DEFAULT_METRICS_PATH,
    ) -> None:
        """
        Initialise the post-session hook.

        Args:
            session_id:   Session identifier for the metrics record.
            metrics_path: Path to the JSONL metrics log file.
        """
        self.session_id = session_id
        self.metrics_path = metrics_path

    def run(self, metrics: Optional[SessionMetricsDict] = None, **kwargs: object) -> HookResult:
        """Spawn a daemon thread to write metrics."""
        metrics = metrics or {}
        thread = threading.Thread(
            target=self._run_async,
            args=(metrics,),
            daemon=True,
            name="post-session-hook",
        )
        thread.start()
        return HookResult(success=True, message="Post-session hook started in background.")

    def _run_async(self, metrics: SessionMetricsDict) -> None:
        """
        Write session metrics on the background thread.

        Args:
            metrics: Session metrics dict from ``SessionMetrics.to_dict()``.
        """
        logger.debug("post_session: starting async tasks", extra={"data": {"session_id": self.session_id}})
        self._write_metrics(metrics)
        logger.debug("post_session: done", extra={"data": {"session_id": self.session_id}})

    def _write_metrics(self, metrics: SessionMetricsDict) -> None:
        """
        Append a session metrics record to the JSONL metrics log.

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
