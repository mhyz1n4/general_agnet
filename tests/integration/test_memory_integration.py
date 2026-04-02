"""
Integration tests for the memory system.

These tests use a real Redis instance on port 6380 and a real filesystem.
LLM interactions are not used here.
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


@pytest.fixture
def mem_root(tmp_path):
    return str(tmp_path / "memory")


@pytest.fixture
def full_memory_manager(mem_root):
    storage = TypedMarkdownStorage(memory_root=mem_root)
    indexer = JSONIndexer(index_path=os.path.join(mem_root, "index.json"))
    retriever = KeywordRetriever(indexer=indexer)
    return MemoryManager(storage=storage, indexer=indexer, retriever=retriever)


@pytest.fixture
def redis_memory_manager(mem_root, clean_redis):
    from src.memory.redis.storage import RedisStorage
    storage = TypedMarkdownStorage(memory_root=mem_root)
    indexer = JSONIndexer(index_path=os.path.join(mem_root, "index.json"))
    retriever = KeywordRetriever(indexer=indexer)
    session_storage = RedisStorage(host="localhost", port=6380, default_ttl=60)
    return MemoryManager(
        storage=storage,
        indexer=indexer,
        retriever=retriever,
        session_storage=session_storage,
    )


# ---------------------------------------------------------------------------
# Basic save & retrieval
# ---------------------------------------------------------------------------

def test_save_and_retrieve(full_memory_manager):
    full_memory_manager.save_message(
        "k1", "Python is a programming language", {"type": "semantic"}
    )
    full_memory_manager.save_message(
        "k2", "I like coffee in the morning", {"type": "semantic", "topic": "personal"}
    )
    full_memory_manager.save_message(
        "k3", "Use pytest for Python testing", {"type": "procedural"}
    )

    ctx = full_memory_manager.get_context("Python programming")
    assert "Python" in ctx or "programming" in ctx


def test_retrieval_ranks_by_relevance(full_memory_manager):
    full_memory_manager.save_message("r1", "Python is great", {"type": "semantic"})
    full_memory_manager.save_message("r2", "Python testing with pytest", {"type": "procedural"})
    full_memory_manager.save_message("r3", "coffee is delicious", {"type": "semantic"})

    ctx, keys = full_memory_manager.get_context_with_keys("Python", limit=2)
    assert len(keys) <= 2
    assert any("r1" == k or "r2" == k for k in keys)


# ---------------------------------------------------------------------------
# Redis session mirroring
# ---------------------------------------------------------------------------

def test_redis_session_mirroring(redis_memory_manager, clean_redis):
    redis_memory_manager.save_message(
        "session_msg1", "Remember this fact", {"type": "semantic"}
    )
    # Check key exists in Redis with TTL
    raw = clean_redis.get("mem:session_msg1")
    assert raw is not None
    ttl = clean_redis.ttl("mem:session_msg1")
    assert ttl > 0


# ---------------------------------------------------------------------------
# Typed directory structure
# ---------------------------------------------------------------------------

def test_episodic_file_in_correct_dir(mem_root, full_memory_manager):
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m")
    full_memory_manager.save_message(
        "ep1",
        "Today we had a team meeting",
        {"type": "episodic", "timestamp": datetime.now(timezone.utc).isoformat()},
    )
    conv_dir = os.path.join(mem_root, "conversations", ts)
    assert os.path.isfile(os.path.join(conv_dir, "ep1.md"))


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_dedup_prevents_double_write(mem_root, full_memory_manager):
    content = "Unique content that must not be duplicated"
    full_memory_manager.save_message("d1", content, {"type": "semantic"})
    full_memory_manager.save_message("d2", content, {"type": "semantic"})

    # Only one .md file should exist
    md_files = []
    for root, dirs, files in os.walk(mem_root):
        for f in files:
            if f.endswith(".md"):
                md_files.append(f)
    assert md_files.count("d2.md") == 0


# ---------------------------------------------------------------------------
# DLQ retry
# ---------------------------------------------------------------------------

def test_dlq_retry_on_storage_failure(tmp_path):
    import stat
    mem_root = str(tmp_path / "mem")
    os.makedirs(mem_root)

    storage = TypedMarkdownStorage(memory_root=mem_root)
    indexer = JSONIndexer(index_path=os.path.join(mem_root, "index.json"))
    retriever = KeywordRetriever(indexer=indexer)
    dlq = DeadLetterQueue(dlq_path=str(tmp_path / "dlq.jsonl"), max_attempts=3)
    mm = MemoryManager(
        storage=storage, indexer=indexer, retriever=retriever, dlq=dlq
    )

    # Make storage temporarily unwritable by patching save
    original_save = storage.save
    storage.save = lambda *a, **kw: (_ for _ in ()).throw(OSError("injected failure"))

    try:
        mm.save_message("fail1", "should go to dlq", {"type": "semantic"})
    except OSError:
        pass

    assert len(dlq._load_entries()) == 1

    # Restore and retry
    storage.save = original_save
    count = dlq.retry_all(mm)
    assert count == 1
    assert len(dlq._load_entries()) == 0
