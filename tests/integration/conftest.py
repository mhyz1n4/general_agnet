"""
Shared fixtures and test data for integration tests.

Fixtures
--------
mem_root     — isolated tmp directory for each test's memory files
mem_stack    — full memory stack (storage + indexer + retriever + manager)
               with a save_message spy; used by orchestrator and memory tests
session_id   — unique session ID per test
strands_patch — patches Strands SDK to handle vLLM streaming quirks (#815)
agent_or_mock — real Strands Agent when LLM is reachable, else a MagicMock

Test data
---------
TEST_MEMORIES — list of canonical memory records shared across integration tests;
                covers all three memory types (episodic, semantic, procedural).
"""

import os
import uuid
from unittest.mock import MagicMock

import pytest

from src.constants import SESSION_ID_HEX_LENGTH
from src.memory.file_system.indexer import JSONIndexer
from src.memory.file_system.retriever import KeywordRetriever
from src.memory.file_system.typed_storage import TypedMarkdownStorage
from src.memory.manager import MemoryManager


def pytest_configure(config):
    """Register the ``integration`` marker so pytest does not warn about it."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require Redis + FS)"
    )


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

#: Canonical memory records used by integration tests.
#: Each entry maps directly to ``MemoryManager.save_message(key, content, metadata)``.
TEST_MEMORIES = [
    {
        "key": "semantic_python",
        "content": "Python is a programming language",
        "metadata": {"type": "semantic", "topic": "programming"},
    },
    {
        "key": "semantic_coffee",
        "content": "I like coffee in the morning",
        "metadata": {"type": "semantic", "topic": "personal"},
    },
    {
        "key": "procedural_pytest",
        "content": "Use pytest for Python testing",
        "metadata": {"type": "procedural", "topic": "testing"},
    },
    {
        "key": "episodic_meeting",
        "content": "Today we had a team meeting",
        "metadata": {"type": "episodic", "session_id": "test-session"},
    },
    # Used by test_orchestrator_integration — keywords must overlap with query
    {
        "key": "semantic_dark_mode",
        "content": "User prefers dark mode in editors",
        "metadata": {"type": "semantic", "topic": "preferences"},
    },
]


# ---------------------------------------------------------------------------
# Test data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def TEST_MEMORIES():
    """Return the shared test memory records as a fixture for parametric tests."""
    return [
        {
            "key": "semantic_python",
            "content": "Python is a programming language",
            "metadata": {"type": "semantic", "topic": "programming"},
        },
        {
            "key": "semantic_coffee",
            "content": "I like coffee in the morning",
            "metadata": {"type": "semantic", "topic": "personal"},
        },
        {
            "key": "procedural_pytest",
            "content": "Use pytest for Python testing",
            "metadata": {"type": "procedural", "topic": "testing"},
        },
        {
            "key": "episodic_meeting",
            "content": "Today we had a team meeting",
            "metadata": {"type": "episodic", "session_id": "test-session"},
        },
        {
            "key": "semantic_dark_mode",
            "content": "User prefers dark mode in editors",
            "metadata": {"type": "semantic", "topic": "preferences"},
        },
    ]


# ---------------------------------------------------------------------------
# Core memory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_root(tmp_path):
    """Isolated filesystem root for a single test's memory files."""
    return str(tmp_path / "memory")


@pytest.fixture
def mem_stack(mem_root):
    """
    Build a complete, isolated memory stack wired to ``mem_root``.

    Wraps ``MemoryManager.save_message`` with a spy so tests can assert on
    every write without touching the underlying implementation.

    Returns a dict with:
      ``root``       — memory root path
      ``index_path`` — path to ``index.json``
      ``storage``    — ``TypedMarkdownStorage`` instance
      ``indexer``    — ``JSONIndexer`` instance
      ``retriever``  — ``KeywordRetriever`` instance
      ``manager``    — ``MemoryManager`` instance
      ``spy_calls``  — list of ``{message_id, content, metadata}`` dicts
    """
    index_path = os.path.join(mem_root, "index.json")
    storage = TypedMarkdownStorage(memory_root=mem_root)
    indexer = JSONIndexer(index_path=index_path)
    retriever = KeywordRetriever(index_path=index_path, storage=storage)
    manager = MemoryManager(storage=storage, indexer=indexer, retriever=retriever)

    spy_calls: list[dict] = []
    _original_save = manager.save_message

    def _spy(message_id, content, metadata=None):
        """Record each save and delegate to the real implementation."""
        spy_calls.append(
            {"message_id": message_id, "content": content, "metadata": metadata or {}}
        )
        return _original_save(message_id, content, metadata)

    manager.save_message = _spy

    return {
        "root": mem_root,
        "index_path": index_path,
        "storage": storage,
        "indexer": indexer,
        "retriever": retriever,
        "manager": manager,
        "spy_calls": spy_calls,
    }


@pytest.fixture
def session_id() -> str:
    """Unique session ID per test run, formatted like runtime IDs."""
    return f"test_{uuid.uuid4().hex[:SESSION_ID_HEX_LENGTH]}"


# ---------------------------------------------------------------------------
# Strands SDK patch fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def strands_patch():
    """
    Patch Strands SDK to handle vLLM streaming quirks (strands issue #815).

    vLLM streaming sends the first tool-call chunk with ``function.name=None``,
    which causes ``validate_tool_use_name`` to call ``re.match(pattern, None)``
    and raise a ``TypeError``.  This patch converts that into the expected
    ``InvalidToolUseNameException`` so Strands can handle it gracefully.

    Also enforces the per-invocation tool-call limit stored in
    ``src.orchestrator.tool_call_counter``.

    Restores the original function after each test.
    """
    try:
        import strands.tools.tools as _strands_tools
        import strands.event_loop.streaming as _strands_streaming
        from strands.tools.tools import InvalidToolUseNameException
        from src.orchestrator import tool_call_counter

        _orig = _strands_tools.validate_tool_use_name

        def _safe(tool: dict) -> None:
            """Guarded replacement for ``validate_tool_use_name``."""
            count = getattr(tool_call_counter, "count", 0) + 1
            limit = getattr(tool_call_counter, "limit", 0)
            tool_call_counter.count = count
            if limit and count > limit:
                raise InvalidToolUseNameException(
                    f"tool call limit reached ({count}/{limit})"
                )
            if not tool.get("name"):
                raise InvalidToolUseNameException(
                    "tool name is None or empty (strands #815)"
                )
            _orig(tool)

        _strands_tools.validate_tool_use_name = _safe
        _strands_streaming.validate_tool_use_name = _safe
        yield
        _strands_tools.validate_tool_use_name = _orig
        _strands_streaming.validate_tool_use_name = _orig
    except ImportError:
        # strands not installed; LLM tests will be skipped via llm_available
        yield


# ---------------------------------------------------------------------------
# Agent fixture — real or mock depending on LLM availability
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_or_mock(llm_endpoint_or_none, strands_patch):
    """
    Return a real Strands Agent when the LLM endpoint is reachable, else a
    ``MagicMock`` that returns a fixed response string.

    Tests that use this fixture run against a live model when available, so
    they exercise the full inference path without requiring a manual skip.

    The returned object is always callable as ``agent(prompt) -> str``.
    """
    if llm_endpoint_or_none is None:
        return MagicMock(return_value="This is the assistant response.")

    try:
        from strands import Agent
        from strands.models.openai import OpenAIModel
    except ImportError:
        return MagicMock(return_value="This is the assistant response.")

    from src.config import Config
    from src.orchestrator import tool_call_counter

    config = Config()
    tool_call_counter.count = 0
    tool_call_counter.limit = config.max_tool_calls

    model = OpenAIModel(
        client_args={
            "api_key": config.llm_api_key,
            "base_url": config.llm_api_endpoint,
        },
        model_id=config.llm_model,
        # Use 4096 tokens so Qwen3 can finish its <think> block plus response
        # without hitting MaxTokensReachedException during orchestrator tests.
        params={"max_tokens": max(config.llm_max_tokens, 4096)},
    )
    return Agent(model=model, tools=[], callback_handler=None)
