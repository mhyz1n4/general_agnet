"""
Unit tests for KeywordRetriever.

Verifies compound and individual-token search, relevance scoring, the
fallback search path, limit enforcement, and graceful handling of storage
misses and empty inputs.  Storage and the index file are both mocked so
no filesystem I/O is required.
"""

import json
import pytest
from typing import Dict
from unittest.mock import MagicMock, patch

from src.memory.file_system.retriever import KeywordRetriever
from src.memory.base import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_storage() -> MagicMock:
    """Mock BaseStorage whose load() returns a simple StorageRecord by default."""
    storage = MagicMock()
    storage.load.side_effect = lambda key: {"id": key, "content": f"content of {key}", "metadata": {}}
    return storage


def _make_retriever(index_data: Dict, storage: MagicMock) -> KeywordRetriever:
    """
    Build a KeywordRetriever whose _load_index() returns *index_data* directly.

    Patches ``open`` so no real file is read.
    """
    retriever = KeywordRetriever(index_path="/fake/index.json", storage=storage)
    with patch.object(retriever, "_load_index", return_value=index_data):
        pass  # patch applied via return_value on the instance below
    retriever._load_index = MagicMock(return_value=index_data)
    return retriever


# ---------------------------------------------------------------------------
# Compound-query search
# ---------------------------------------------------------------------------


class TestCompoundSearch:
    """Compound (all-token) search path: every token must appear in the entry."""

    def test_multi_keyword_match_returns_result(self, mock_storage: MagicMock) -> None:
        """An entry whose keywords include all query tokens should be returned."""
        index = {"k1": {"keywords": ["python", "preference", "editor"], "metadata": {}}}
        retriever = _make_retriever(index, mock_storage)
        results = retriever.search("python preference")
        assert len(results) == 1
        assert results[0].key == "k1"

    def test_higher_score_ranks_first(self, mock_storage: MagicMock) -> None:
        """Entry with more matching keywords should rank above one with fewer."""
        index = {
            "k1": {"keywords": ["python", "editor", "preference"], "metadata": {}},
            "k2": {"keywords": ["python"], "metadata": {}},
        }
        retriever = _make_retriever(index, mock_storage)
        results = retriever.search("python editor preference")
        assert results[0].key == "k1"

    def test_no_keyword_match_returns_empty(self, mock_storage: MagicMock) -> None:
        """A query with no matching keywords should return an empty list."""
        index = {"k1": {"keywords": ["golang", "channels"], "metadata": {}}}
        retriever = _make_retriever(index, mock_storage)
        results = retriever.search("python preference")
        # Compound search fails; fallback tries individual tokens but still no match.
        assert results == []

    def test_limit_respected(self, mock_storage: MagicMock) -> None:
        """Results should be capped at *limit* even if more entries match."""
        index = {f"k{i}": {"keywords": ["python"], "metadata": {}} for i in range(10)}
        retriever = _make_retriever(index, mock_storage)
        results = retriever.search("python", limit=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Fallback (individual-token) search
# ---------------------------------------------------------------------------


class TestFallbackSearch:
    """When the compound search returns nothing, individual tokens are tried."""

    def test_fallback_triggered_on_no_compound_match(self, mock_storage: MagicMock) -> None:
        """
        Entry matching only one token should appear via fallback when compound search fails.
        The index has 'python' but not both 'python' and 'unknown', so compound fails.
        """
        index = {"k1": {"keywords": ["python"], "metadata": {}}}
        retriever = _make_retriever(index, mock_storage)
        results = retriever.search("python unknown_token")
        assert any(r.key == "k1" for r in results)

    def test_fallback_deduplicates_results(self, mock_storage: MagicMock) -> None:
        """An entry matching multiple individual tokens should appear only once."""
        index = {"k1": {"keywords": ["python", "editor"], "metadata": {}}}
        retriever = _make_retriever(index, mock_storage)
        results = retriever.search("python editor unknown")
        keys = [r.key for r in results]
        assert keys.count("k1") == 1


# ---------------------------------------------------------------------------
# Storage miss handling
# ---------------------------------------------------------------------------


class TestStorageMiss:
    """Entries whose storage.load() returns None should be silently skipped."""

    def test_none_from_storage_skipped(self, mock_storage: MagicMock) -> None:
        """If storage.load() returns None, the entry is excluded from results."""
        # side_effect takes precedence over return_value on MagicMock; clear both.
        mock_storage.load.side_effect = None
        mock_storage.load.return_value = None
        index = {"k1": {"keywords": ["python"], "metadata": {}}}
        retriever = _make_retriever(index, mock_storage)
        results = retriever.search("python")
        assert results == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions that must not raise exceptions."""

    def test_empty_query_returns_empty(self, mock_storage: MagicMock) -> None:
        """An empty or whitespace-only query should return an empty list immediately."""
        index = {"k1": {"keywords": ["python"], "metadata": {}}}
        retriever = _make_retriever(index, mock_storage)
        assert retriever.search("") == []
        assert retriever.search("   ") == []

    def test_empty_index_returns_empty(self, mock_storage: MagicMock) -> None:
        """A query against an empty index should return an empty list."""
        retriever = _make_retriever({}, mock_storage)
        assert retriever.search("python") == []

    def test_score_is_float(self, mock_storage: MagicMock) -> None:
        """relevance_score on SearchResult must be a float as per the datamodel."""
        index = {"k1": {"keywords": ["python"], "metadata": {}}}
        retriever = _make_retriever(index, mock_storage)
        results = retriever.search("python")
        assert isinstance(results[0].relevance_score, float)
