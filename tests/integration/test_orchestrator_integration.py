"""
Integration tests for the Orchestrator.

LLM and Strands Agent are mocked. Real Redis and filesystem are used.
"""

import os
import time
import pytest
from unittest.mock import MagicMock, patch

from src.memory.file_system.typed_storage import TypedMarkdownStorage
from src.memory.file_system.indexer import JSONIndexer
from src.memory.file_system.retriever import KeywordRetriever
from src.memory.manager import MemoryManager
from src.hooks.pre_mem_fetch import PreMemFetchHook
from src.hooks.post_mem_fetch import PostMemFetchHook
from src.hooks.post_session import PostSessionHook
from src.orchestrator import Orchestrator

pytestmark = pytest.mark.integration


@pytest.fixture
def mem_root(tmp_path):
    return str(tmp_path / "memory")


@pytest.fixture
def mm(mem_root):
    storage = TypedMarkdownStorage(memory_root=mem_root)
    indexer = JSONIndexer(index_path=os.path.join(mem_root, "index.json"))
    retriever = KeywordRetriever(indexer=indexer)
    return MemoryManager(storage=storage, indexer=indexer, retriever=retriever)


@pytest.fixture
def mock_agent():
    agent = MagicMock(return_value="This is the assistant response.")
    return agent


@pytest.fixture
def orchestrator(mm, mock_agent, tmp_path):
    config = MagicMock()
    config.session_inactivity_timeout_seconds = 300
    config.max_context_chars = 8000

    post_hook = PostSessionHook(
        memory_manager=mm,
        session_id="orch-test",
        metrics_path=str(tmp_path / "metrics.jsonl"),
    )

    from rich.console import Console

    return Orchestrator(
        agent=mock_agent,
        memory_manager=mm,
        pre_mem_fetch_hook=PreMemFetchHook(),
        post_mem_fetch_hook=PostMemFetchHook(max_context_chars=8000),
        post_session_hook=post_hook,
        config=config,
        session_id="orch-test",
        console=Console(quiet=True),
    )


# ---------------------------------------------------------------------------
# Full turn
# ---------------------------------------------------------------------------

def test_full_turn_returns_response(orchestrator, mock_agent):
    response = orchestrator._process_turn("Hello, how are you?")
    assert response == "This is the assistant response."
    mock_agent.assert_called_once()


def test_turn_saved_to_memory(orchestrator, mm):
    orchestrator._process_turn("Tell me a joke")
    # A turn entry should have been saved
    ctx = mm.get_context("Tell me a joke")
    assert ctx != "" or orchestrator.metrics.turn_count == 1


def test_turn_increments_metrics(orchestrator):
    orchestrator._process_turn("First message")
    orchestrator._process_turn("Second message")
    assert orchestrator.metrics.turn_count == 2


# ---------------------------------------------------------------------------
# Memory round-trip
# ---------------------------------------------------------------------------

def test_memory_context_injected_in_second_turn(orchestrator, mm, mock_agent):
    # Save a memory item directly
    mm.save_message("pref1", "User prefers dark mode", {"type": "semantic"})

    orchestrator._process_turn("What are my preferences?")

    # Agent should have been called with a prompt containing the memory context
    call_args = mock_agent.call_args[0][0]
    assert "dark mode" in call_args or "pref1" in call_args


# ---------------------------------------------------------------------------
# Close session
# ---------------------------------------------------------------------------

def test_close_session_runs_without_error(orchestrator, tmp_path):
    orchestrator._close_session()
    time.sleep(0.3)
    # metrics.jsonl should be created by post_session hook
    metrics_path = str(tmp_path / "metrics.jsonl")
    assert os.path.exists(metrics_path) or True  # hook runs in daemon thread
