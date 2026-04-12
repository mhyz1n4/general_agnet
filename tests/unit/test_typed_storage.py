"""
Unit tests for TypedMarkdownStorage.

Verifies directory routing per memory type, load/save round-trips, manifest
integrity, overwrite behaviour, concurrent write safety, delete, list_keys,
exists, and filesystem-safe topic sanitisation.
All tests use tmp_path so no real project files are touched.
"""

import os
import threading
from pathlib import Path
from typing import Optional

import pytest

from src.memory.file_system.typed_storage import TypedMarkdownStorage
from src.memory.types import StorageRecord


@pytest.fixture
def storage(tmp_path: Path) -> TypedMarkdownStorage:
    """Fresh TypedMarkdownStorage backed by a unique temp directory."""
    return TypedMarkdownStorage(memory_root=str(tmp_path))


def _save(
    storage: TypedMarkdownStorage,
    key: str,
    content: str,
    mem_type: str,
    topic: Optional[str] = None,
    timestamp: Optional[str] = None,
    session_id: Optional[str] = None,
) -> StorageRecord:
    """
    Helper that builds a StorageRecord and persists it via storage.save().

    Returns the record dict so callers can inspect it if needed.
    """
    meta: dict = {"type": mem_type}
    if topic:
        meta["topic"] = topic
    if timestamp:
        meta["timestamp"] = timestamp
    if session_id:
        meta["session_id"] = session_id
    data: StorageRecord = {"id": key, "content": content, "metadata": meta}
    storage.save(key, data)
    return data


# ---------------------------------------------------------------------------
# Directory routing
# ---------------------------------------------------------------------------


def test_episodic_stored_under_session_and_date(storage: TypedMarkdownStorage, tmp_path: Path) -> None:
    """Episodic entries must be routed to conversations/{session_id}/{YYYY-MM-DD}/{key}.md."""
    _save(
        storage, "ep1", "hello", "episodic",
        session_id="abc123",
        timestamp="2026-03-15T10:00:00+00:00",
    )
    assert os.path.exists(
        os.path.join(str(tmp_path), "conversations", "abc123", "2026-03-15", "ep1.md")
    )


def test_episodic_missing_session_id_uses_unknown(storage: TypedMarkdownStorage, tmp_path: Path) -> None:
    """An episodic entry without a session_id must fall back to the 'unknown_session' directory."""
    _save(storage, "ep2", "no session", "episodic", timestamp="2026-03-15T10:00:00+00:00")
    assert os.path.exists(
        os.path.join(str(tmp_path), "conversations", "unknown_session", "2026-03-15", "ep2.md")
    )


def test_episodic_different_sessions_stored_separately(storage: TypedMarkdownStorage, tmp_path: Path) -> None:
    """Two episodic entries from different sessions must be stored in separate directories."""
    _save(storage, "ep3", "session A", "episodic", session_id="sess_a", timestamp="2026-03-15T10:00:00+00:00")
    _save(storage, "ep4", "session B", "episodic", session_id="sess_b", timestamp="2026-03-15T10:00:00+00:00")
    assert os.path.exists(os.path.join(str(tmp_path), "conversations", "sess_a", "2026-03-15", "ep3.md"))
    assert os.path.exists(os.path.join(str(tmp_path), "conversations", "sess_b", "2026-03-15", "ep4.md"))


def test_episodic_same_session_different_days(storage: TypedMarkdownStorage, tmp_path: Path) -> None:
    """Episodic entries from the same session but different dates must be in separate date dirs."""
    _save(storage, "ep5", "day one", "episodic", session_id="sess_x", timestamp="2026-03-15T10:00:00+00:00")
    _save(storage, "ep6", "day two", "episodic", session_id="sess_x", timestamp="2026-03-16T10:00:00+00:00")
    assert os.path.exists(os.path.join(str(tmp_path), "conversations", "sess_x", "2026-03-15", "ep5.md"))
    assert os.path.exists(os.path.join(str(tmp_path), "conversations", "sess_x", "2026-03-16", "ep6.md"))


def test_semantic_with_topic(storage: TypedMarkdownStorage, tmp_path: Path) -> None:
    """Semantic entries with a topic are stored under knowledge/{topic}/{key}.md."""
    _save(storage, "sem1", "I like Python", "semantic", topic="preferences")
    assert os.path.exists(os.path.join(str(tmp_path), "knowledge", "preferences", "sem1.md"))


def test_semantic_without_topic_uses_general(storage: TypedMarkdownStorage, tmp_path: Path) -> None:
    """Semantic entries without a topic default to knowledge/general/{key}.md."""
    _save(storage, "sem2", "fact", "semantic")
    assert os.path.exists(os.path.join(str(tmp_path), "knowledge", "general", "sem2.md"))


def test_procedural_routing(storage: TypedMarkdownStorage, tmp_path: Path) -> None:
    """Procedural entries are stored under procedures/{topic}/{key}.md."""
    _save(storage, "proc1", "steps", "procedural", topic="setup")
    assert os.path.exists(os.path.join(str(tmp_path), "procedures", "setup", "proc1.md"))


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_existing_key(storage: TypedMarkdownStorage) -> None:
    """load() must return the stored content with the correct id."""
    _save(storage, "k1", "content value", "semantic")
    result: Optional[StorageRecord] = storage.load("k1")
    assert result is not None
    assert result["id"] == "k1"
    assert result["content"] == "content value"


def test_load_missing_key_returns_none(storage: TypedMarkdownStorage) -> None:
    """load() for a key that was never saved must return None."""
    assert storage.load("nonexistent") is None


def test_load_after_manual_delete_returns_none(storage: TypedMarkdownStorage, tmp_path: Path) -> None:
    """load() must return None when the .md file has been deleted behind storage's back."""
    _save(storage, "del1", "data", "semantic")
    manifest: dict = storage._load_manifest()
    rel_path: str = manifest["del1"]
    abs_path: str = os.path.join(str(tmp_path), rel_path)
    os.unlink(abs_path)
    assert storage.load("del1") is None


# ---------------------------------------------------------------------------
# Overwrite, manifest integrity
# ---------------------------------------------------------------------------


def test_same_key_saved_twice_overwrites(storage: TypedMarkdownStorage) -> None:
    """Saving the same key twice must overwrite the content with only one manifest entry."""
    _save(storage, "ow1", "original", "semantic")
    _save(storage, "ow1", "updated", "semantic")
    result: Optional[StorageRecord] = storage.load("ow1")
    assert result is not None
    assert result["content"] == "updated"
    manifest: dict = storage._load_manifest()
    assert list(manifest.keys()).count("ow1") == 1


# ---------------------------------------------------------------------------
# Markdown round-trip
# ---------------------------------------------------------------------------


def test_content_with_markdown_preserved(storage: TypedMarkdownStorage) -> None:
    """Content containing markdown syntax must survive the save/load round-trip intact."""
    content: str = "# Header\n\n- item 1\n- item 2\n\n**bold**"
    _save(storage, "md1", content, "semantic")
    result: Optional[StorageRecord] = storage.load("md1")
    assert result is not None
    assert result["content"] == content


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_saves_no_manifest_corruption(storage: TypedMarkdownStorage) -> None:
    """20 threads saving distinct keys concurrently must all appear in the manifest."""
    errors: list = []

    def save_one(i: int) -> None:
        """Worker: save one entry under a unique key."""
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
    manifest: dict = storage._load_manifest()
    for i in range(20):
        assert f"c{i}" in manifest


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


def test_delete_existing_key(storage: TypedMarkdownStorage) -> None:
    """delete() on a saved key must return True and make load() return None."""
    _save(storage, "d1", "bye", "episodic")
    assert storage.delete("d1") is True
    assert storage.load("d1") is None


def test_delete_missing_key_returns_false(storage: TypedMarkdownStorage) -> None:
    """delete() on a key that was never saved must return False without raising."""
    assert storage.delete("no_such_key") is False


# ---------------------------------------------------------------------------
# list_keys()
# ---------------------------------------------------------------------------


def test_list_keys_returns_all_saved_keys(storage: TypedMarkdownStorage) -> None:
    """list_keys() must return every key that has been saved, across all memory types."""
    _save(storage, "ep1", "content", "episodic", session_id="s1", timestamp="2026-03-15T10:00:00+00:00")
    _save(storage, "sem1", "content", "semantic", topic="work")
    _save(storage, "proc1", "content", "procedural", topic="setup")
    keys = storage.list_keys()
    assert set(keys) == {"ep1", "sem1", "proc1"}


def test_list_keys_empty_on_fresh_storage(storage: TypedMarkdownStorage) -> None:
    """A freshly created storage with no saves must return an empty list from list_keys()."""
    assert storage.list_keys() == []


def test_list_keys_excludes_deleted_keys(storage: TypedMarkdownStorage) -> None:
    """A key removed via delete() must not appear in list_keys()."""
    _save(storage, "k1", "content", "semantic")
    _save(storage, "k2", "content", "semantic")
    storage.delete("k1")
    assert "k1" not in storage.list_keys()
    assert "k2" in storage.list_keys()


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------


def test_exists_true_for_saved_key(storage: TypedMarkdownStorage) -> None:
    """exists() must return True for a key that has been saved."""
    _save(storage, "e1", "content", "semantic")
    assert storage.exists("e1") is True


def test_exists_false_for_missing_key(storage: TypedMarkdownStorage) -> None:
    """exists() must return False for a key that has never been saved."""
    assert storage.exists("never_saved") is False


def test_exists_false_after_delete(storage: TypedMarkdownStorage) -> None:
    """exists() must return False for a key that was saved and then deleted."""
    _save(storage, "e2", "content", "semantic")
    storage.delete("e2")
    assert storage.exists("e2") is False


# ---------------------------------------------------------------------------
# Topic sanitisation in path
# ---------------------------------------------------------------------------


def test_topic_with_spaces_sanitised_in_path(storage: TypedMarkdownStorage, tmp_path: Path) -> None:
    """Spaces in a topic name must be replaced so the path remains filesystem-safe."""
    _save(storage, "t1", "content", "semantic", topic="my work project")
    # The sanitised name replaces spaces with '_'
    sanitised_topic: str = "my_work_project"
    assert os.path.exists(
        os.path.join(str(tmp_path), "knowledge", sanitised_topic, "t1.md")
    )
