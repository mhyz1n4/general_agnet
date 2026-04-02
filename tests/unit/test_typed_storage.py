"""Unit tests for TypedMarkdownStorage."""

import os
import threading
import pytest
from src.memory.file_system.typed_storage import TypedMarkdownStorage


@pytest.fixture
def storage(tmp_path):
    return TypedMarkdownStorage(memory_root=str(tmp_path))


def _save(storage, key, content, mem_type, topic=None, timestamp=None):
    meta = {"type": mem_type}
    if topic:
        meta["topic"] = topic
    if timestamp:
        meta["timestamp"] = timestamp
    data = {"id": key, "content": content, "metadata": meta}
    storage.save(key, data)
    return data


# ---------------------------------------------------------------------------
# Directory routing
# ---------------------------------------------------------------------------

def test_episodic_stored_in_conversations_yyyymm(storage, tmp_path):
    _save(storage, "ep1", "hello", "episodic", timestamp="2026-03-15T10:00:00+00:00")
    assert os.path.exists(os.path.join(str(tmp_path), "conversations", "2026-03", "ep1.md"))


def test_semantic_with_topic(storage, tmp_path):
    _save(storage, "sem1", "I like Python", "semantic", topic="preferences")
    assert os.path.exists(os.path.join(str(tmp_path), "knowledge", "preferences", "sem1.md"))


def test_semantic_without_topic_uses_general(storage, tmp_path):
    _save(storage, "sem2", "fact", "semantic")
    assert os.path.exists(os.path.join(str(tmp_path), "knowledge", "general", "sem2.md"))


def test_procedural_routing(storage, tmp_path):
    _save(storage, "proc1", "steps", "procedural", topic="setup")
    assert os.path.exists(os.path.join(str(tmp_path), "procedures", "setup", "proc1.md"))


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

def test_load_existing_key(storage):
    _save(storage, "k1", "content value", "semantic")
    result = storage.load("k1")
    assert result is not None
    assert result["id"] == "k1"
    assert result["content"] == "content value"


def test_load_missing_key_returns_none(storage):
    assert storage.load("nonexistent") is None


def test_load_after_manual_delete_returns_none(storage, tmp_path):
    _save(storage, "del1", "data", "semantic")
    # Manually delete the file
    manifest = storage._load_manifest()
    rel_path = manifest["del1"]
    abs_path = os.path.join(str(tmp_path), rel_path)
    os.unlink(abs_path)
    assert storage.load("del1") is None


# ---------------------------------------------------------------------------
# Overwrite, manifest integrity
# ---------------------------------------------------------------------------

def test_same_key_saved_twice_overwrites(storage):
    _save(storage, "ow1", "original", "semantic")
    _save(storage, "ow1", "updated", "semantic")
    result = storage.load("ow1")
    assert result["content"] == "updated"
    # Only one manifest entry
    manifest = storage._load_manifest()
    assert list(manifest.keys()).count("ow1") == 1


# ---------------------------------------------------------------------------
# Markdown round-trip
# ---------------------------------------------------------------------------

def test_content_with_markdown_preserved(storage):
    content = "# Header\n\n- item 1\n- item 2\n\n**bold**"
    _save(storage, "md1", content, "semantic")
    result = storage.load("md1")
    assert result["content"] == content


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_saves_no_manifest_corruption(storage):
    errors = []

    def save_one(i):
        try:
            _save(storage, f"c{i}", f"content {i}", "semantic")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=save_one, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    manifest = storage._load_manifest()
    # All 20 keys should be present
    for i in range(20):
        assert f"c{i}" in manifest


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

def test_delete_existing_key(storage):
    _save(storage, "d1", "bye", "episodic")
    assert storage.delete("d1") is True
    assert storage.load("d1") is None


def test_delete_missing_key_returns_false(storage):
    assert storage.delete("no_such_key") is False
