"""
Integration tests for hooks against real Redis and filesystem.

LLM client is mocked throughout; only Redis and the filesystem are real.
Start Redis on port 6380 with ``make test-infra-start`` before running.
"""

import json
import os
import time
import pytest
from unittest.mock import MagicMock

from src.hooks.pre_session import PreSessionHook
from src.hooks.post_session import PostSessionHook
from src.memory.file_system.typed_storage import TypedMarkdownStorage
from src.memory.file_system.indexer import JSONIndexer
from src.memory.file_system.retriever import KeywordRetriever
from src.memory.manager import MemoryManager

pytestmark = pytest.mark.integration

REDIS_TEST_PORT = 6380


# mem_root and mem_stack come from tests/integration/conftest.py


@pytest.fixture
def full_mm(mem_stack):
    """
    MemoryManager from the shared mem_stack, alias for hooks tests.

    Using the shared ``mem_stack`` fixture avoids duplicating the
    storage/indexer/retriever setup across multiple test files.
    """
    return mem_stack["manager"]


# ---------------------------------------------------------------------------
# PreSessionHook
# ---------------------------------------------------------------------------

def test_pre_session_all_healthy(mem_root, clean_redis):
    """PreSessionHook reports success and no degradation when Redis is up."""
    import redis
    hook = PreSessionHook(
        memory_root=mem_root,
        redis_client=redis.Redis(port=REDIS_TEST_PORT, decode_responses=True),
    )
    result = hook.run()
    assert result.success is True
    assert result.degraded is False


def test_pre_session_redis_wrong_port(mem_root):
    """PreSessionHook degrades gracefully when Redis is unreachable."""
    import redis
    hook = PreSessionHook(
        memory_root=mem_root,
        redis_client=redis.Redis(port=9999),
    )
    result = hook.run()
    assert result.success is True
    assert result.degraded is True
    assert "Redis" in result.message


# ---------------------------------------------------------------------------
# PostSessionHook — flush Redis → FS
# ---------------------------------------------------------------------------

def test_post_session_flushes_redis_to_fs(mem_root, clean_redis, tmp_path):
    """Messages pre-populated in Redis must appear as .md files after hook runs."""
    from datetime import datetime, timezone

    redis_client = clean_redis  # already flushed, live connection on port 6380

    # Pre-populate Redis with 3 messages
    for i in range(3):
        msg_id = f"msg_{i}"
        data = {
            "id": msg_id,
            "content": f"session message {i}",
            "metadata": {
                "type": "episodic",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
        redis_client.set(f"mem:{msg_id}", json.dumps(data), ex=60)

    storage = TypedMarkdownStorage(memory_root=mem_root)
    indexer = JSONIndexer(index_path=os.path.join(mem_root, "index.json"))
    retriever = KeywordRetriever(index_path=os.path.join(mem_root, "index.json"), storage=storage)
    mm = MemoryManager(storage=storage, indexer=indexer, retriever=retriever)

    mock_llm = MagicMock()
    mock_llm.completion.return_value = MagicMock(
        content=[MagicMock(text="Session summary text")]
    )

    hook = PostSessionHook(
        memory_manager=mm,
        session_id="int-test-session",
        redis_client=redis_client,
        redis_prefix="mem:",
        llm_client=mock_llm,
        metrics_path=str(tmp_path / "metrics.jsonl"),
    )
    hook.run(metrics={"turns": [{"user": "hi", "assistant": "hello"}]})

    # Give daemon thread time to complete
    time.sleep(1.0)

    md_files = []
    for root, dirs, files in os.walk(mem_root):
        for f in files:
            if f.endswith(".md"):
                md_files.append(f)

    # Expect at least the 3 flushed messages
    assert len(md_files) >= 3


def test_post_session_llm_mock_summary_saved(mem_root, clean_redis, tmp_path):
    """PostSessionHook must persist the LLM-generated session summary to disk."""
    redis_client = clean_redis

    storage = TypedMarkdownStorage(memory_root=mem_root)
    indexer = JSONIndexer(index_path=os.path.join(mem_root, "index.json"))
    retriever = KeywordRetriever(index_path=os.path.join(mem_root, "index.json"), storage=storage)
    mm = MemoryManager(storage=storage, indexer=indexer, retriever=retriever)

    mock_llm = MagicMock()
    mock_llm.completion.return_value = MagicMock(
        content=[MagicMock(text="A great session today.")]
    )

    hook = PostSessionHook(
        memory_manager=mm,
        session_id="sum-session",
        redis_client=redis_client,
        llm_client=mock_llm,
        metrics_path=str(tmp_path / "metrics.jsonl"),
    )
    hook.run(metrics={"turns": [{"user": "hello", "assistant": "hi"}]})
    time.sleep(0.5)

    # Summary must be written to conversations/ as session_<id>.md
    conv_dir = os.path.join(mem_root, "conversations")
    found = any(
        "session_sum-session.md" in files
        for _, _, files in os.walk(conv_dir)
    )
    assert found
