"""
Unit tests for DeadLetterQueue.

Covers enqueue, retry_all success/failure/dead-entry paths, persistence
across restarts, and thread-safety under concurrent enqueue calls.
All tests use tmp_path so no real project files are touched.
"""

import json
import os
import threading
from pathlib import Path
from typing import List

import pytest
from unittest.mock import MagicMock

from src.memory.dlq import DeadLetterQueue, DLQEntry


@pytest.fixture
def dlq(tmp_path: Path) -> DeadLetterQueue:
    """Fresh DeadLetterQueue backed by a temp JSONL file with max 3 retry attempts."""
    return DeadLetterQueue(dlq_path=str(tmp_path / "dlq.jsonl"), max_attempts=3)


@pytest.fixture
def mock_mm() -> MagicMock:
    """Mock MemoryManager whose save_message succeeds by default."""
    return MagicMock()


# ---------------------------------------------------------------------------
# enqueue()
# ---------------------------------------------------------------------------


def test_enqueue_creates_entry(dlq: DeadLetterQueue) -> None:
    """Enqueuing one message creates exactly one entry with attempts=0 and dead=False."""
    dlq.enqueue("msg1", "content", {"type": "semantic"}, "OSError: disk full")
    entries: List[DLQEntry] = dlq._load_entries()
    assert len(entries) == 1
    assert entries[0].message_id == "msg1"
    assert entries[0].attempts == 0
    assert entries[0].dead is False


def test_enqueue_multiple_entries(dlq: DeadLetterQueue) -> None:
    """Enqueuing two distinct messages creates two separate entries."""
    dlq.enqueue("m1", "c1", {}, "err1")
    dlq.enqueue("m2", "c2", {}, "err2")
    assert len(dlq._load_entries()) == 2


# ---------------------------------------------------------------------------
# retry_all()
# ---------------------------------------------------------------------------


def test_retry_all_success_removes_entry(dlq: DeadLetterQueue, mock_mm: MagicMock) -> None:
    """A successful retry removes the entry from the DLQ and returns count=1."""
    dlq.enqueue("msg1", "content", {}, "error")
    count: int = dlq.retry_all(mock_mm)
    assert count == 1
    assert dlq._load_entries() == []
    mock_mm.save_message.assert_called_once_with("msg1", "content", {})


def test_retry_all_failure_increments_attempts(dlq: DeadLetterQueue, mock_mm: MagicMock) -> None:
    """A failed retry increments attempts by 1 but does not mark the entry dead (below max)."""
    mock_mm.save_message.side_effect = RuntimeError("still failing")
    dlq.enqueue("msg1", "content", {}, "original error")
    dlq.retry_all(mock_mm)
    entries: List[DLQEntry] = dlq._load_entries()
    assert entries[0].attempts == 1
    assert entries[0].dead is False


def test_retry_all_marks_dead_after_max_attempts(dlq: DeadLetterQueue, mock_mm: MagicMock) -> None:
    """After max_attempts failed retries the entry is marked dead and no longer retried."""
    mock_mm.save_message.side_effect = RuntimeError("fail")
    dlq.enqueue("msg1", "content", {}, "err")
    for _ in range(3):
        dlq.retry_all(mock_mm)
    entries: List[DLQEntry] = dlq._load_entries()
    assert entries[0].dead is True


def test_retry_all_dead_entry_not_retried(dlq: DeadLetterQueue, mock_mm: MagicMock) -> None:
    """An entry already marked dead must be skipped; save_message must not be called."""
    dlq.enqueue("msg1", "content", {}, "err")
    entries: List[DLQEntry] = dlq._load_entries()
    entries[0].dead = True
    dlq._save_entries(entries)

    dlq.retry_all(mock_mm)
    mock_mm.save_message.assert_not_called()


def test_retry_all_mixed_entries(dlq: DeadLetterQueue, mock_mm: MagicMock) -> None:
    """When one entry succeeds and one fails, only the failed entry remains in the DLQ."""
    def side_effect(msg_id: str, content: str, meta: dict) -> None:
        """Raise only for the 'bad' message ID."""
        if msg_id == "bad":
            raise RuntimeError("fail")

    mock_mm.save_message.side_effect = side_effect
    dlq.enqueue("good", "c1", {}, "err")
    dlq.enqueue("bad", "c2", {}, "err")

    count: int = dlq.retry_all(mock_mm)
    assert count == 1
    remaining: List[DLQEntry] = dlq._load_entries()
    assert len(remaining) == 1
    assert remaining[0].message_id == "bad"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_dlq_survives_restart(tmp_path: Path) -> None:
    """Entries written by one DLQ instance must be readable by a fresh instance on the same path."""
    path: str = str(tmp_path / "dlq.jsonl")
    dlq1: DeadLetterQueue = DeadLetterQueue(dlq_path=path, max_attempts=3)
    dlq1.enqueue("msg1", "content", {}, "err")

    dlq2: DeadLetterQueue = DeadLetterQueue(dlq_path=path, max_attempts=3)
    entries: List[DLQEntry] = dlq2._load_entries()
    assert len(entries) == 1
    assert entries[0].message_id == "msg1"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_enqueue_no_corruption(dlq: DeadLetterQueue) -> None:
    """20 threads each enqueuing a unique message should all appear in the final DLQ file."""
    errors: list = []

    def enqueue_one(i: int) -> None:
        """Worker: enqueue one entry."""
        try:
            dlq.enqueue(f"m{i}", f"content {i}", {}, f"err {i}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=enqueue_one, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(dlq._load_entries()) == 20
