"""
Latency benchmarks for the memory system and orchestrator turn overhead.

Design targets (§10c):
  - Memory retrieval (search): P50 < 100ms, P95 < 100ms
  - Full orchestrator turn overhead (excl. LLM latency): P50 < 200ms, P95 < 200ms

Run with:
    pytest tests/benchmarks/ -v -m benchmark

These tests use a real filesystem (tmp_path) but mock the LLM agent call so that
LLM latency is excluded from the turn-overhead measurement.
"""

import shutil
import statistics
import tempfile
import time
from typing import List
from unittest.mock import MagicMock

import pytest

from src.memory.file_system.indexer import JSONIndexer
from src.memory.file_system.retriever import KeywordRetriever
from src.memory.file_system.storage import FileStorage
from src.memory.manager import MemoryManager


pytestmark = pytest.mark.benchmark

_ITERATIONS = 50   # number of timed repetitions per benchmark
_P50_TARGET_MS = 100
_P95_TARGET_MS = 100


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def seeded_memory():
    """
    Create a temporary memory system with 100 pre-indexed entries.
    Shared across all benchmarks in this module.
    """
    tmpdir = tempfile.mkdtemp(prefix="bench_")
    storage_path = f"{tmpdir}/storage"
    index_path = f"{tmpdir}/index.json"

    storage = FileStorage(storage_path)
    indexer = JSONIndexer(index_path)
    retriever = KeywordRetriever(index_path=index_path, storage=storage)
    mm = MemoryManager(storage=storage, indexer=indexer, retriever=retriever)

    words = [
        "python", "memory", "agent", "search", "retrieval", "knowledge",
        "procedure", "conversation", "session", "context", "episodic",
        "semantic", "tool", "hook", "config", "redis", "index", "storage",
    ]
    for i in range(100):
        word_a = words[i % len(words)]
        word_b = words[(i + 3) % len(words)]
        mm.save_message(
            f"entry_{i:03d}",
            f"Entry {i}: {word_a} and {word_b} are important concepts.",
            {"type": "semantic", "topic": word_a},
        )

    yield mm, retriever

    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _measure(fn, iterations: int = _ITERATIONS) -> List[float]:
    """Return a list of wall-clock milliseconds for `fn()` called `iterations` times."""
    samples: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def _p50(samples: List[float]) -> float:
    return statistics.median(samples)


def _p95(samples: List[float]) -> float:
    sorted_s = sorted(samples)
    idx = max(0, int(len(sorted_s) * 0.95) - 1)
    return sorted_s[idx]


# ---------------------------------------------------------------------------
# Benchmark: memory retrieval
# ---------------------------------------------------------------------------


def test_retrieval_p50_under_100ms(seeded_memory):
    """P50 of KeywordRetriever.search() must be < 100ms."""
    mm, retriever = seeded_memory
    samples = _measure(lambda: retriever.search("python memory agent", limit=5))
    p50 = _p50(samples)
    print(f"\n[retrieval] P50={p50:.1f}ms  P95={_p95(samples):.1f}ms  (n={len(samples)})")
    assert p50 < _P50_TARGET_MS, f"P50 {p50:.1f}ms exceeds {_P50_TARGET_MS}ms target"


def test_retrieval_p95_under_100ms(seeded_memory):
    """P95 of KeywordRetriever.search() must be < 100ms."""
    mm, retriever = seeded_memory
    samples = _measure(lambda: retriever.search("knowledge context retrieval", limit=5))
    p95 = _p95(samples)
    print(f"\n[retrieval] P95={p95:.1f}ms  (n={len(samples)})")
    assert p95 < _P95_TARGET_MS, f"P95 {p95:.1f}ms exceeds {_P95_TARGET_MS}ms target"


def test_retrieval_fallback_p50_under_100ms(seeded_memory):
    """P50 of fallback retrieval (individual tokens) must be < 100ms."""
    mm, retriever = seeded_memory
    # "zephyr" won't match anything — triggers single-token fallback on each word
    samples = _measure(lambda: retriever.search("zephyr python memory", limit=5))
    p50 = _p50(samples)
    print(f"\n[retrieval-fallback] P50={p50:.1f}ms  (n={len(samples)})")
    assert p50 < _P50_TARGET_MS


# ---------------------------------------------------------------------------
# Benchmark: MemoryManager.get_context() overhead
# ---------------------------------------------------------------------------


def test_get_context_p50_under_100ms(seeded_memory):
    """P50 of MemoryManager.get_context() (incl. classify + retrieve) < 100ms."""
    mm, _ = seeded_memory
    samples = _measure(lambda: mm.get_context("python memory agent", limit=3))
    p50 = _p50(samples)
    print(f"\n[get_context] P50={p50:.1f}ms  P95={_p95(samples):.1f}ms")
    assert p50 < _P50_TARGET_MS


# ---------------------------------------------------------------------------
# Benchmark: orchestrator turn overhead (excl. LLM)
# ---------------------------------------------------------------------------


def test_orchestrator_turn_overhead_p50_under_200ms(seeded_memory, tmp_path):
    """
    P50 of orchestrator._process_turn() with agent() mocked must be < 200ms.
    This measures: pre-hook + retrieval + post-hook + prompt render + save_message.
    """
    from src.config import Config
    from src.hooks.post_mem_fetch import PostMemFetchHook
    from src.hooks.post_session import PostSessionHook
    from src.hooks.pre_mem_fetch import PreMemFetchHook
    from src.orchestrator import Orchestrator

    mm, _ = seeded_memory

    mock_agent = MagicMock(return_value="mock response")

    cfg = MagicMock(spec=Config)
    cfg.session_inactivity_timeout_seconds = 300
    cfg.tool_timeout_seconds = 30
    cfg.max_context_chars = 8000
    cfg.max_tool_calls = 10
    cfg.llm_max_tokens = 1024
    cfg.index_path = str(tmp_path / "index.json")
    cfg.memory_root = str(tmp_path / "memory")

    post_session = MagicMock(spec=PostSessionHook)

    orch = Orchestrator(
        agent=mock_agent,
        memory_manager=mm,
        pre_mem_fetch_hook=PreMemFetchHook(),
        post_mem_fetch_hook=PostMemFetchHook(max_context_chars=8000),
        post_session_hook=post_session,
        config=cfg,
        session_id="bench_session",
    )

    samples = _measure(lambda: orch._process_turn("what do you know about python?"))
    p50 = _p50(samples)
    p95 = _p95(samples)
    print(f"\n[turn_overhead] P50={p50:.1f}ms  P95={p95:.1f}ms  (n={len(samples)})")
    assert p50 < 200, f"P50 {p50:.1f}ms exceeds 200ms target"


def test_orchestrator_turn_overhead_p95_under_200ms(seeded_memory, tmp_path):
    """P95 of orchestrator._process_turn() must be < 200ms."""
    from src.config import Config
    from src.hooks.post_mem_fetch import PostMemFetchHook
    from src.hooks.post_session import PostSessionHook
    from src.hooks.pre_mem_fetch import PreMemFetchHook
    from src.orchestrator import Orchestrator

    mm, _ = seeded_memory
    mock_agent = MagicMock(return_value="mock response")

    cfg = MagicMock(spec=Config)
    cfg.session_inactivity_timeout_seconds = 300
    cfg.tool_timeout_seconds = 30
    cfg.max_context_chars = 8000
    cfg.max_tool_calls = 10
    cfg.llm_max_tokens = 1024
    cfg.index_path = str(tmp_path / "index.json")
    cfg.memory_root = str(tmp_path / "memory")

    orch = Orchestrator(
        agent=mock_agent,
        memory_manager=mm,
        pre_mem_fetch_hook=PreMemFetchHook(),
        post_mem_fetch_hook=PostMemFetchHook(max_context_chars=8000),
        post_session_hook=MagicMock(spec=PostSessionHook),
        config=cfg,
        session_id="bench_session_2",
    )

    samples = _measure(lambda: orch._process_turn("how do I deploy to production?"))
    p95 = _p95(samples)
    print(f"\n[turn_overhead] P95={p95:.1f}ms  (n={len(samples)})")
    assert p95 < 200, f"P95 {p95:.1f}ms exceeds 200ms target"
