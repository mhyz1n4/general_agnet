"""
Smoke tests for the memory system (FileStorage + JSONIndexer + KeywordRetriever).

These tests use the flat ``FileStorage`` backend (not ``TypedMarkdownStorage``)
to exercise the lowest-level save/retrieve path.  Each test gets an isolated
directory via pytest's ``tmp_path`` fixture — no files are ever written to the
project working directory.
"""

import json
import os

import pytest

from src.memory.file_system.storage import FileStorage
from src.memory.file_system.indexer import JSONIndexer
from src.memory.file_system.retriever import KeywordRetriever
from src.memory.manager import MemoryManager


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def setup_memory_system(tmp_path):
    """
    Create an isolated memory stack under ``tmp_path``.

    Yields a ``MemoryManager`` backed by ``FileStorage`` so tests can call
    ``save_message`` / ``get_context`` directly.  The directory is cleaned up
    automatically by pytest after each test.
    """
    storage_path = str(tmp_path / "storage")
    index_path = str(tmp_path / "index.json")

    storage = FileStorage(storage_path)
    indexer = JSONIndexer(index_path)
    retriever = KeywordRetriever(index_path, storage)
    manager = MemoryManager(storage, indexer, retriever)

    yield manager


# ---------------------------------------------------------------------------
# Basic save & retrieval
# ---------------------------------------------------------------------------

def test_save_and_retrieve(setup_memory_system):
    """Content saved under a key must be returned when queried by keyword."""
    manager = setup_memory_system
    manager.save_message("msg1", "The quick brown fox jumps over the lazy dog.", {"type": "test"})

    context = manager.get_context("quick fox")
    assert "quick brown fox" in context


def test_multiple_messages_ranking(setup_memory_system):
    """Keyword search must return relevant messages and exclude unrelated ones."""
    manager = setup_memory_system
    manager.save_message("msg1", "Python is a great programming language.", {"lang": "python"})
    manager.save_message("msg2", "Python is versatile and easy to learn.", {"lang": "python"})
    manager.save_message("msg3", "JavaScript is used for web development.", {"lang": "js"})

    context = manager.get_context("Python")
    assert "versatile" in context
    assert "programming language" in context
    assert "JavaScript" not in context


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_edge_case_no_results(setup_memory_system):
    """Query with no matching keywords must return an empty string."""
    manager = setup_memory_system
    context = manager.get_context("nonexistent word")
    assert context == ""


def test_edge_case_empty_query(setup_memory_system):
    """Empty query must return an empty string without crashing."""
    manager = setup_memory_system
    context = manager.get_context("")
    assert context == ""


def test_edge_case_special_chars(setup_memory_system):
    """
    Queries containing only special characters must not crash.

    ``KeywordRetriever`` strips non-word tokens so the result may be empty;
    the important invariant is that no exception is raised.
    """
    manager = setup_memory_system
    manager.save_message("msg_special", "Special characters like !@#$%^&*() should be handled.", {})
    context = manager.get_context("!@#$%^&*()")
    assert isinstance(context, str)


def test_corrupt_index_file(setup_memory_system, tmp_path):
    """A corrupted index.json must be handled gracefully (empty result, no crash)."""
    manager = setup_memory_system
    manager.save_message("msg1", "Normal message", {})

    index_path = str(tmp_path / "index.json")
    with open(index_path, "w") as f:
        f.write("{ invalid json }")

    context = manager.get_context("Normal")
    assert context == ""


def test_file_missing_from_storage(setup_memory_system, tmp_path):
    """
    A key present in the index but missing on disk must be handled gracefully.

    The retriever should skip the missing file and return an empty context
    rather than raising ``FileNotFoundError``.
    """
    manager = setup_memory_system
    manager.save_message("msg1", "I exist in index but will be deleted from disk", {})

    storage_path = str(tmp_path / "storage")
    os.remove(os.path.join(storage_path, "msg1.json"))

    context = manager.get_context("index")
    assert context == ""


# ---------------------------------------------------------------------------
# Retriever: individual-token fallback
# ---------------------------------------------------------------------------

def test_retriever_fallback_individual_tokens(setup_memory_system):
    """
    A compound query with no exact match must retry on individual tokens.

    "Python rocks" misses on the compound form; falling back to "Python" alone
    should still find the saved record.
    """
    manager = setup_memory_system
    manager.save_message("r1", "Python is awesome", {"type": "semantic"})

    context = manager.get_context("Python rocks")
    assert "Python is awesome" in context


def test_retriever_no_fallback_when_single_token(setup_memory_system):
    """A single-token query with no match must return empty (no infinite loop)."""
    manager = setup_memory_system
    manager.save_message("r2", "Go is fast", {"type": "semantic"})
    context = manager.get_context("nonexistentword")
    assert context == ""


def test_retriever_fallback_deduplicates_results(setup_memory_system):
    """
    A record matching multiple individual tokens from the fallback path must
    appear only once in the returned context.
    """
    manager = setup_memory_system
    manager.save_message("r3", "Python Go Rust are all great languages", {"type": "semantic"})
    # Each of Python, Go, Rust individually matches r3 — ensure no duplication
    context = manager.get_context("Python Go Rust")
    # The content (not the internal key) should appear exactly once
    assert context.count("Python Go Rust are all great languages") == 1
