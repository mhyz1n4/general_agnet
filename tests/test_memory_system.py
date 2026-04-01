import os
import shutil
import pytest
import json
from src.memory.file_system.storage import FileStorage
from src.memory.file_system.indexer import JSONIndexer
from src.memory.file_system.retriever import KeywordRetriever
from src.memory.manager import MemoryManager

TEST_BASE_PATH = "test_memory_data"
TEST_STORAGE_PATH = os.path.join(TEST_BASE_PATH, "storage")
TEST_INDEX_PATH = os.path.join(TEST_BASE_PATH, "index.json")

@pytest.fixture
def setup_memory_system():
    """Setup and teardown for test memory system."""
    if os.path.exists(TEST_BASE_PATH):
        shutil.rmtree(TEST_BASE_PATH)
    os.makedirs(TEST_STORAGE_PATH)

    storage = FileStorage(TEST_STORAGE_PATH)
    indexer = JSONIndexer(TEST_INDEX_PATH)
    retriever = KeywordRetriever(TEST_INDEX_PATH, storage)
    manager = MemoryManager(storage, indexer, retriever)

    yield manager

    # Teardown
    if os.path.exists(TEST_BASE_PATH):
        shutil.rmtree(TEST_BASE_PATH)

def test_save_and_retrieve(setup_memory_system):
    manager = setup_memory_system
    manager.save_message("msg1", "The quick brown fox jumps over the lazy dog.", {"type": "test"})

    # Test retrieval by keyword
    context = manager.get_context("quick fox")
    assert "quick brown fox" in context
    assert "--- Context (Key: msg1) ---" in context

def test_multiple_messages_ranking(setup_memory_system):
    manager = setup_memory_system
    manager.save_message("msg1", "Python is a great programming language.", {"lang": "python"})
    manager.save_message("msg2", "Python is versatile and easy to learn.", {"lang": "python"})
    manager.save_message("msg3", "JavaScript is used for web development.", {"lang": "js"})

    # Search for "Python" should return msg1 and msg2
    context = manager.get_context("Python")
    assert "versatile" in context
    assert "programming language" in context
    assert "JavaScript" not in context

def test_edge_case_no_results(setup_memory_system):
    manager = setup_memory_system
    context = manager.get_context("nonexistent word")
    assert context == ""

def test_edge_case_empty_query(setup_memory_system):
    manager = setup_memory_system
    context = manager.get_context("")
    assert context == ""

def test_edge_case_special_chars(setup_memory_system):
    manager = setup_memory_system
    manager.save_message("msg_special", "Special characters like !@#$%^&*() should be handled.", {})
    context = manager.get_context("!@#$%^&*()")
    # Currently KeywordRetriever splits by whitespace, so this might return empty
    # This test confirms no crash happens
    assert isinstance(context, str)

def test_corrupt_index_file(setup_memory_system):
    manager = setup_memory_system
    manager.save_message("msg1", "Normal message", {})

    # Corrupt the index file
    with open(TEST_INDEX_PATH, "w") as f:
        f.write("{ invalid json }")

    # The indexer/retriever should handle this gracefully (log error, return empty/reset)
    context = manager.get_context("Normal")
    # KeywordRetriever currently returns {} on exception
    assert context == ""

def test_file_missing_from_storage(setup_memory_system):
    manager = setup_memory_system
    manager.save_message("msg1", "I exist in index but will be deleted from disk", {})

    # Delete the actual storage file but keep it in index
    os.remove(os.path.join(TEST_STORAGE_PATH, "msg1.json"))

    # Retrieval should handle FileNotFoundError gracefully
    context = manager.get_context("index")
    assert context == ""
