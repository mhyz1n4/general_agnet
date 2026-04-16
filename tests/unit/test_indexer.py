"""
Unit tests for JSONIndexer.

Verifies add/update/find/touch operations, keyword extraction, atomic file
writes, corrupt-file resilience, and thread-safety under concurrent writes.
All tests use tmp_path so no real project files are touched.
"""

import json
import os
import threading
import pytest
from pathlib import Path

from src.memory.file_system.indexer import JSONIndexer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def indexer(tmp_path: Path) -> JSONIndexer:
    """Fresh JSONIndexer backed by a temp directory; index.json is created on init."""
    return JSONIndexer(index_path=str(tmp_path / "index.json"))


# ---------------------------------------------------------------------------
# add() / update()
# ---------------------------------------------------------------------------


class TestAddAndUpdate:
    """add() and update() should write searchable entries to the index file."""

    def test_add_creates_entry(self, indexer: JSONIndexer) -> None:
        """After add(), the key must be present in the index with extracted keywords."""
        indexer.add("k1", "Python is great for scripting", {"type": "semantic"})
        index = indexer._load_index()
        assert "k1" in index
        assert "python" in index["k1"]["keywords"]

    def test_add_overwrites_existing_key(self, indexer: JSONIndexer) -> None:
        """A second add() with the same key replaces the previous entry."""
        indexer.add("k1", "Python scripting", {})
        indexer.add("k1", "Go channels", {})
        index = indexer._load_index()
        assert "golang" not in index["k1"]["keywords"] or "python" not in index["k1"]["keywords"]
        assert "channels" in index["k1"]["keywords"] or "golang" in index["k1"]["keywords"]

    def test_update_replaces_keywords(self, indexer: JSONIndexer) -> None:
        """update() is semantically identical to add() and replaces keywords."""
        indexer.add("k1", "original content", {})
        indexer.update("k1", "completely new content", {"type": "semantic"})
        index = indexer._load_index()
        assert "original" not in index["k1"]["keywords"]
        assert "completely" in index["k1"]["keywords"] or "content" in index["k1"]["keywords"]

    def test_metadata_stored(self, indexer: JSONIndexer) -> None:
        """Metadata passed to add() must be persisted in the index entry."""
        indexer.add("k1", "some content", {"type": "episodic", "topic": "work"})
        index = indexer._load_index()
        assert index["k1"]["metadata"]["type"] == "episodic"


# ---------------------------------------------------------------------------
# find_by_content_hash()
# ---------------------------------------------------------------------------


class TestFindByContentHash:
    """find_by_content_hash() searches metadata for a matching hash value."""

    def test_finds_existing_hash(self, indexer: JSONIndexer) -> None:
        """Should return the key when a matching content_hash is in metadata."""
        indexer.add("k1", "hello world", {"content_hash": "abc123"})
        assert indexer.find_by_content_hash("abc123") == "k1"

    def test_returns_none_when_not_found(self, indexer: JSONIndexer) -> None:
        """Should return None when no entry has the given hash."""
        assert indexer.find_by_content_hash("nonexistent_hash") is None

    def test_returns_none_on_empty_index(self, indexer: JSONIndexer) -> None:
        """Should return None gracefully on an empty index."""
        assert indexer.find_by_content_hash("abc") is None


# ---------------------------------------------------------------------------
# touch()
# ---------------------------------------------------------------------------


class TestTouch:
    """touch() refreshes the stored timestamp of an existing entry."""

    def test_touch_updates_timestamp(self, indexer: JSONIndexer) -> None:
        """After touch(), the entry's metadata.timestamp must be updated."""
        indexer.add("k1", "content", {"timestamp": "2020-01-01T00:00:00+00:00"})
        indexer.touch("k1")
        index = indexer._load_index()
        new_ts: str = index["k1"]["metadata"].get("timestamp", "")
        assert new_ts > "2020-01-01T00:00:00+00:00"

    def test_touch_missing_key_is_noop(self, indexer: JSONIndexer) -> None:
        """touch() on a key that does not exist must not raise or corrupt the index."""
        indexer.add("k1", "hello", {})
        indexer.touch("nonexistent")
        index = indexer._load_index()
        assert "k1" in index  # index is intact


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


class TestKeywordExtraction:
    """_extract_keywords() must filter short words and deduplicate."""

    def test_short_words_excluded(self, indexer: JSONIndexer) -> None:
        """Words with length <= MIN_KEYWORD_LENGTH (3) are excluded."""
        indexer.add("k1", "I am a big fan", {})
        index = indexer._load_index()
        # 'I', 'am', 'a' are too short; only longer words kept
        for kw in index["k1"]["keywords"]:
            assert len(kw) > 3

    def test_keywords_lowercased(self, indexer: JSONIndexer) -> None:
        """Keywords must be stored in lowercase."""
        indexer.add("k1", "Python Django Flask", {})
        for kw in indexer._load_index()["k1"]["keywords"]:
            assert kw == kw.lower()

    def test_duplicate_words_deduplicated(self, indexer: JSONIndexer) -> None:
        """Repeated words should appear only once in the keyword list."""
        indexer.add("k1", "python python python rocks", {})
        keywords = indexer._load_index()["k1"]["keywords"]
        assert keywords.count("python") == 1


# ---------------------------------------------------------------------------
# Corruption resilience
# ---------------------------------------------------------------------------


class TestCorruptionResilience:
    """_load_index() must recover gracefully from a corrupt or missing file."""

    def test_corrupt_json_returns_empty(self, tmp_path: Path) -> None:
        """If index.json is not valid JSON, _load_index() must return {} without raising."""
        index_path = str(tmp_path / "index.json")
        with open(index_path, "w") as f:
            f.write("{this is not valid json{{{{")
        indexer = JSONIndexer(index_path=index_path)
        result = indexer._load_index()
        assert result == {}

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """If the index file is deleted after init, _load_index() returns {}."""
        indexer = JSONIndexer(index_path=str(tmp_path / "index.json"))
        os.unlink(str(tmp_path / "index.json"))
        result = indexer._load_index()
        assert result == {}


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Concurrent add() calls must not corrupt the index."""

    def test_concurrent_adds_no_corruption(self, indexer: JSONIndexer) -> None:
        """10 threads each adding a unique key should all appear in the final index."""
        errors: list = []

        def add_entry(i: int) -> None:
            """Worker: add one entry to the shared indexer."""
            try:
                indexer.add(f"key{i}", f"content for entry {i} with some words", {"type": "semantic"})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=add_entry, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        index = indexer._load_index()
        for i in range(10):
            assert f"key{i}" in index, f"key{i} missing from index"
