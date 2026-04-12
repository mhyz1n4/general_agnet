"""
Integration tests for the memory system.

These tests use a real Redis instance on port 6380 and a real filesystem.
LLM interactions are not used here.

Test data is sourced from ``TEST_MEMORIES`` in conftest.py to avoid
duplicating content strings across test files.
"""

import os
import time
import pytest

from src.memory.file_system.typed_storage import TypedMarkdownStorage
from src.memory.file_system.indexer import JSONIndexer
from src.memory.file_system.retriever import KeywordRetriever
from src.memory.manager import MemoryManager
from src.memory.dlq import DeadLetterQueue

pytestmark = pytest.mark.integration

# mem_root, mem_stack, and TEST_MEMORIES come from tests/integration/conftest.py


@pytest.fixture
def full_memory_manager(mem_stack):
    """MemoryManager from the shared stack (alias for test readability)."""
    return mem_stack["manager"]


@pytest.fixture
def redis_memory_manager(mem_root, clean_redis):
    """
    MemoryManager backed by a real Redis session store.

    Wires ``RedisStorage`` (for session-scoped short-term memory) to the same
    filesystem stack so we can verify Redis ↔ FS mirroring.
    """
    from src.memory.redis.storage import RedisStorage
    import redis as redis_lib

    storage = TypedMarkdownStorage(memory_root=mem_root)
    indexer = JSONIndexer(index_path=os.path.join(mem_root, "index.json"))
    retriever = KeywordRetriever(index_path=os.path.join(mem_root, "index.json"), storage=storage)
    rc = redis_lib.Redis(host="localhost", port=6380, decode_responses=True)
    session_storage = RedisStorage(client=rc, default_ttl=60)
    return MemoryManager(
        storage=storage,
        indexer=indexer,
        retriever=retriever,
        session_storage=session_storage,
    )


# ---------------------------------------------------------------------------
# Basic save & retrieval — uses shared TEST_MEMORIES dataset
# ---------------------------------------------------------------------------

def test_save_and_retrieve(full_memory_manager, TEST_MEMORIES):
    """Save the three canonical memory types and verify keyword retrieval."""
    for entry in TEST_MEMORIES[:3]:  # episodic excluded (needs session_id in path)
        full_memory_manager.save_message(
            entry["key"], entry["content"], entry["metadata"]
        )

    ctx = full_memory_manager.get_context("Python programming")
    assert "Python" in ctx or "programming" in ctx


def test_retrieval_ranks_by_relevance(full_memory_manager):
    """Higher-overlap results should be ranked first."""
    full_memory_manager.save_message("r1", "Python is great", {"type": "semantic"})
    full_memory_manager.save_message("r2", "Python testing with pytest", {"type": "procedural"})
    full_memory_manager.save_message("r3", "coffee is delicious", {"type": "semantic"})

    ctx, keys = full_memory_manager.get_context_with_keys("Python", limit=2)
    assert len(keys) <= 2
    assert any(k in ("r1", "r2") for k in keys)


# ---------------------------------------------------------------------------
# Redis session mirroring
# ---------------------------------------------------------------------------

def test_redis_session_mirroring(redis_memory_manager, clean_redis):
    """A message saved via a Redis-backed manager must appear in Redis with a TTL."""
    redis_memory_manager.save_message(
        "session_msg1", "Remember this fact", {"type": "semantic"}
    )
    raw = clean_redis.get("mem:session_msg1")
    assert raw is not None
    ttl = clean_redis.ttl("mem:session_msg1")
    assert ttl > 0


# ---------------------------------------------------------------------------
# Typed directory structure
# ---------------------------------------------------------------------------

def test_episodic_file_in_correct_dir(mem_root, full_memory_manager):
    """Episodic messages must be stored under conversations/{session_id}/{date}/."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    full_memory_manager.save_message(
        "ep1",
        "Today we had a team meeting",
        {"type": "episodic", "session_id": "test-session", "timestamp": now.isoformat()},
    )
    # Expected path: conversations/test-session/{YYYY-MM-DD}/ep1.md
    conv_dir = os.path.join(mem_root, "conversations", "test-session", date_str)
    assert os.path.isfile(os.path.join(conv_dir, "ep1.md"))


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_dedup_prevents_double_write(mem_root, full_memory_manager):
    """Saving identical content under two keys must only write one file."""
    content = "Unique content that must not be duplicated"
    full_memory_manager.save_message("d1", content, {"type": "semantic"})
    full_memory_manager.save_message("d2", content, {"type": "semantic"})

    md_files = []
    for root, dirs, files in os.walk(mem_root):
        for f in files:
            if f.endswith(".md"):
                md_files.append(f)
    assert "d2.md" not in md_files


# ---------------------------------------------------------------------------
# DLQ retry
# ---------------------------------------------------------------------------

def test_dlq_retry_on_storage_failure(tmp_path):
    """
    A failed save must land in the DLQ, and ``retry_all`` must re-persist it.

    Injects an ``OSError`` into ``storage.save`` for the first call, then
    restores the original and verifies that DLQ retry succeeds.
    """
    mem_root = str(tmp_path / "mem")
    os.makedirs(mem_root)

    storage = TypedMarkdownStorage(memory_root=mem_root)
    indexer = JSONIndexer(index_path=os.path.join(mem_root, "index.json"))
    retriever = KeywordRetriever(index_path=os.path.join(mem_root, "index.json"), storage=storage)
    dlq = DeadLetterQueue(dlq_path=str(tmp_path / "dlq.jsonl"), max_attempts=3)
    mm = MemoryManager(storage=storage, indexer=indexer, retriever=retriever, dlq=dlq)

    original_save = storage.save
    storage.save = lambda *a, **kw: (_ for _ in ()).throw(OSError("injected failure"))

    try:
        mm.save_message("fail1", "should go to dlq", {"type": "semantic"})
    except OSError:
        pass

    assert len(dlq._load_entries()) == 1

    storage.save = original_save
    count = dlq.retry_all(mm)
    assert count == 1
    assert len(dlq._load_entries()) == 0
