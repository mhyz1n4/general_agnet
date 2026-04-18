"""
Integration tests for hooks against real filesystem.

V1.1: Redis-dependent tests have been retired. Only filesystem and
metrics-writing tests remain.
"""

import json
import os
import time
import pytest
from unittest.mock import MagicMock

from src.hooks.pre_session import PreSessionHook
from src.hooks.post_session import PostSessionHook

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# PreSessionHook
# ---------------------------------------------------------------------------

def test_pre_session_all_healthy(mem_root):
    """PreSessionHook reports success when filesystem is writable."""
    hook = PreSessionHook(memory_root=mem_root)
    result = hook.run()
    assert result.success is True
    assert result.degraded is False


def test_pre_session_creates_missing_dir(tmp_path):
    """PreSessionHook must create memory_root if it does not exist."""
    new_root = str(tmp_path / "brand_new_dir")
    hook = PreSessionHook(memory_root=new_root)
    result = hook.run()
    assert result.success is True
    assert os.path.isdir(new_root)


# ---------------------------------------------------------------------------
# PostSessionHook — metrics writing
# ---------------------------------------------------------------------------

def test_post_session_writes_metrics(tmp_path):
    """PostSessionHook must write a JSONL metrics record."""
    metrics_path = str(tmp_path / "metrics.jsonl")
    hook = PostSessionHook(
        session_id="int-test-session",
        metrics_path=metrics_path,
    )
    hook.run(metrics={"turn_count": 3, "turns": [{"user": "hi", "assistant": "hello"}]})
    time.sleep(0.5)

    assert os.path.exists(metrics_path)
    with open(metrics_path) as f:
        record = json.loads(f.readline())
    assert record["session_id"] == "int-test-session"
    assert record["turn_count"] == 3
    assert "turns" not in record
