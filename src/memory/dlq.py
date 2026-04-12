"""
Dead-Letter Queue (DLQ) for failed memory saves.

When MemoryManager.save_message() fails, the item is persisted to a JSONL file.
A background timer retries failed entries periodically. After dlq_max_attempts
failures the entry is marked dead and left for manual inspection.
"""

import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

from .types import MemoryMetadata
from src.constants import DEFAULT_DLQ_MAX_ATTEMPTS

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .manager import MemoryManager


@dataclass
class DLQEntry:
    """
    One persisted entry in the dead-letter queue.

    Attributes:
        message_id:      Unique identifier of the failed memory save.
        content:         Raw text content that failed to be stored.
        metadata:        Metadata dict that was passed to ``save_message()``.
        error:           String representation of the last exception.
        attempts:        Number of retry attempts made so far.
        last_attempt_ts: ISO-8601 UTC timestamp of the most recent attempt.
        dead:            ``True`` when ``attempts >= max_attempts``; entry
                         will not be retried again and is left for manual
                         inspection.
    """

    message_id: str
    content: str
    metadata: MemoryMetadata
    error: str
    attempts: int = 0
    last_attempt_ts: str = ""
    dead: bool = False


class DeadLetterQueue:
    """Persistent JSONL-backed dead-letter queue with background retry."""

    def __init__(self, dlq_path: str, max_attempts: int = DEFAULT_DLQ_MAX_ATTEMPTS) -> None:
        self.dlq_path = dlq_path
        self.max_attempts = max_attempts
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(dlq_path) if os.path.dirname(dlq_path) else ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        message_id: str,
        content: str,
        metadata: MemoryMetadata,
        error: str,
    ) -> None:
        """Append a failed save to the DLQ."""
        entry = DLQEntry(
            message_id=message_id,
            content=content,
            metadata=metadata,
            error=error,
            last_attempt_ts=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            entries = self._load_entries()
            entries.append(entry)
            self._save_entries(entries)

        logger.error(
            "DLQ: enqueued failed save",
            extra={"data": {"message_id": message_id, "error": error, "alert": True}},
        )

    def retry_all(self, memory_manager: "MemoryManager") -> int:
        """
        Retry all non-dead entries. Returns the number of successfully retried entries.

        Uses a load-outside / merge-inside pattern to avoid two failure modes:

        1. Deadlock: ``save_message`` may itself call ``enqueue``, which tries to
           acquire ``self._lock``.  Holding the lock across ``save_message`` would
           deadlock.

        2. Lost entries: releasing the lock between load and save-back (the old
           pattern) could silently discard any ``enqueue`` calls that occurred
           during processing.

        Solution: snapshot entries outside the lock, process them (calling
        ``save_message`` without holding the lock), then re-acquire the lock and
        merge the results back against the freshly re-read file so any concurrent
        ``enqueue`` calls are preserved.
        """
        with self._lock:
            snapshot: List[DLQEntry] = self._load_entries()

        to_remove: set[str] = set()
        updated_map: Dict[str, DLQEntry] = {}
        retried: int = 0

        for entry in snapshot:
            if entry.dead:
                updated_map[entry.message_id] = entry
                continue

            entry.attempts += 1
            entry.last_attempt_ts = datetime.now(timezone.utc).isoformat()

            try:
                memory_manager.save_message(entry.message_id, entry.content, entry.metadata)
                logger.info(
                    "DLQ: retry succeeded",
                    extra={"data": {"message_id": entry.message_id}},
                )
                to_remove.add(entry.message_id)
                retried += 1
            except Exception as exc:
                entry.error = str(exc)
                if entry.attempts >= self.max_attempts:
                    entry.dead = True
                    logger.error(
                        "DLQ: entry marked dead after max attempts",
                        extra={
                            "data": {
                                "message_id": entry.message_id,
                                "attempts": entry.attempts,
                                "alert": True,
                            }
                        },
                    )
                else:
                    logger.warning(
                        "DLQ: retry failed",
                        extra={
                            "data": {
                                "message_id": entry.message_id,
                                "attempt": entry.attempts,
                                "error": str(exc),
                            }
                        },
                    )
                updated_map[entry.message_id] = entry

        with self._lock:
            # Re-read the file to capture any entries added by enqueue() calls
            # that occurred while we were processing (outside the lock above).
            current: List[DLQEntry] = self._load_entries()
            final: List[DLQEntry] = []
            for e in current:
                if e.message_id in to_remove:
                    continue  # successfully retried — drop from DLQ
                if e.message_id in updated_map:
                    final.append(updated_map[e.message_id])  # use updated state
                else:
                    final.append(e)  # new entry added concurrently — preserve
            self._save_entries(final)

        return retried

    def start_background_retry(
        self, memory_manager: "MemoryManager", interval_seconds: int
    ) -> threading.Event:
        """
        Start a recurring background retry loop using threading.Event.
        Returns the stop_event — call stop_event.set() to stop the loop.
        """
        stop_event = threading.Event()

        def _loop() -> None:
            while not stop_event.wait(interval_seconds):
                try:
                    count = self.retry_all(memory_manager)
                    if count:
                        logger.info("DLQ background retry", extra={"data": {"retried": count}})
                except Exception as exc:
                    logger.error("DLQ background retry error", extra={"data": {"error": str(exc)}})

        thread = threading.Thread(target=_loop, daemon=True, name="dlq-retry")
        thread.start()
        return stop_event

    # ------------------------------------------------------------------
    # Persistence helpers (must be called under self._lock)
    # ------------------------------------------------------------------

    def _load_entries(self) -> List[DLQEntry]:
        """
        Read all entries from the JSONL DLQ file.

        Must be called while ``self._lock`` is held.

        Returns:
            List of ``DLQEntry`` objects; empty list if the file does not
            exist or contains no parseable lines.
        """
        if not os.path.exists(self.dlq_path):
            return []
        entries: List[DLQEntry] = []
        try:
            with open(self.dlq_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(DLQEntry(**json.loads(line)))
                        except (json.JSONDecodeError, TypeError):
                            pass
        except OSError:
            pass
        return entries

    def _save_entries(self, entries: List[DLQEntry]) -> None:
        """
        Atomically overwrite the JSONL DLQ file with the given entries.

        Uses a temp-file + ``os.replace`` to guarantee the file is never
        left in a partially written state.  Must be called while
        ``self._lock`` is held.

        Args:
            entries: Current list of ``DLQEntry`` objects to persist.

        Raises:
            OSError: If the temp file cannot be written or renamed.
        """
        dir_ = os.path.dirname(self.dlq_path) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(asdict(entry)) + "\n")
            os.replace(tmp, self.dlq_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
