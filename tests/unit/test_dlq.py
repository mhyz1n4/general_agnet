"""Unit tests for DeadLetterQueue."""

import json
import os
import threading
import pytest
from unittest.mock import MagicMock, patch

from src.memory.dlq import DeadLetterQueue, DLQEntry


@pytest.fixture
def dlq(tmp_path):
    return DeadLetterQueue(dlq_path=str(tmp_path / "dlq.jsonl"), max_attempts=3)


@pytest.fixture
def mock_mm():
    return MagicMock()


# ---------------------------------------------------------------------------
# enqueue()
# ---------------------------------------------------------------------------

def test_enqueue_creates_entry(dlq):
    dlq.enqueue("msg1", "content", {"type": "semantic"}, "OSError: disk full")
    entries = dlq._load_entries()
    assert len(entries) == 1
    assert entries[0].message_id == "msg1"
    assert entries[0].attempts == 0
    assert entries[0].dead is False


def test_enqueue_multiple_entries(dlq):
    dlq.enqueue("m1", "c1", {}, "err1")
    dlq.enqueue("m2", "c2", {}, "err2")
    assert len(dlq._load_entries()) == 2


# ---------------------------------------------------------------------------
# retry_all()
# ---------------------------------------------------------------------------

def test_retry_all_success_removes_entry(dlq, mock_mm):
    dlq.enqueue("msg1", "content", {}, "error")
    count = dlq.retry_all(mock_mm)
    assert count == 1
    assert dlq._load_entries() == []
    mock_mm.save_message.assert_called_once_with("msg1", "content", {})


def test_retry_all_failure_increments_attempts(dlq, mock_mm):
    mock_mm.save_message.side_effect = RuntimeError("still failing")
    dlq.enqueue("msg1", "content", {}, "original error")
    dlq.retry_all(mock_mm)
    entries = dlq._load_entries()
    assert entries[0].attempts == 1
    assert entries[0].dead is False


def test_retry_all_marks_dead_after_max_attempts(dlq, mock_mm):
    mock_mm.save_message.side_effect = RuntimeError("fail")
    dlq.enqueue("msg1", "content", {}, "err")
    for _ in range(3):
        dlq.retry_all(mock_mm)
    entries = dlq._load_entries()
    assert entries[0].dead is True


def test_retry_all_dead_entry_not_retried(dlq, mock_mm):
    dlq.enqueue("msg1", "content", {}, "err")
    # Mark dead manually
    entries = dlq._load_entries()
    entries[0].dead = True
    dlq._save_entries(entries)

    dlq.retry_all(mock_mm)
    mock_mm.save_message.assert_not_called()


def test_retry_all_mixed_entries(dlq, mock_mm):
    def side_effect(msg_id, content, meta):
        if msg_id == "bad":
            raise RuntimeError("fail")

    mock_mm.save_message.side_effect = side_effect
    dlq.enqueue("good", "c1", {}, "err")
    dlq.enqueue("bad", "c2", {}, "err")

    count = dlq.retry_all(mock_mm)
    assert count == 1
    remaining = dlq._load_entries()
    assert len(remaining) == 1
    assert remaining[0].message_id == "bad"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_dlq_survives_restart(tmp_path):
    path = str(tmp_path / "dlq.jsonl")
    dlq1 = DeadLetterQueue(dlq_path=path, max_attempts=3)
    dlq1.enqueue("msg1", "content", {}, "err")

    # Simulate restart with new instance
    dlq2 = DeadLetterQueue(dlq_path=path, max_attempts=3)
    entries = dlq2._load_entries()
    assert len(entries) == 1
    assert entries[0].message_id == "msg1"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_enqueue_no_corruption(dlq):
    errors = []

    def enqueue_one(i):
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
