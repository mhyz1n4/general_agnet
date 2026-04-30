"""
Latency benchmarks for the memory system and orchestrator turn overhead.

V1.1: Uses StubMemoryProvider (in-memory) for benchmarks. These measure
orchestrator overhead excluding LLM latency.

Design targets (SS10c):
  - Full orchestrator turn overhead (excl. LLM latency): P50 < 200ms, P95 < 200ms

Run with:
    pytest tests/benchmarks/ -v -m benchmark
"""

import statistics
import time
from typing import List
from unittest.mock import MagicMock

import pytest

from src.memory.stub_provider import StubMemoryProvider
from src.memory.provider import SearchFilters

pytestmark = pytest.mark.benchmark

_ITERATIONS = 50
_P50_TARGET_MS = 100
_P95_TARGET_MS = 100


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def seeded_provider():
    """Create a StubMemoryProvider with 100 pre-seeded entries."""
    provider = StubMemoryProvider()
    words = [
        "python", "memory", "agent", "search", "retrieval", "knowledge",
        "procedure", "conversation", "session", "context", "episodic",
        "semantic", "tool", "hook", "config", "redis", "index", "storage",
    ]
    for i in range(100):
        word_a = words[i % len(words)]
        word_b = words[(i + 3) % len(words)]
        provider.save(
            f"Entry {i}: {word_a} and {word_b} are important concepts.",
            type="semantic",
            topic=word_a,
        )
    return provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _measure(fn, iterations: int = _ITERATIONS) -> List[float]:
    """Return a list of wall-clock milliseconds for fn() called iterations times."""
    samples: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def _p50(samples: List[float]) -> float:
    """Return the median of the samples."""
    return statistics.median(samples)


def _p95(samples: List[float]) -> float:
    """Return the 95th percentile of the samples."""
    sorted_s = sorted(samples)
    idx = max(0, int(len(sorted_s) * 0.95) - 1)
    return sorted_s[idx]


# ---------------------------------------------------------------------------
# Benchmark: memory search
# ---------------------------------------------------------------------------


def test_search_p50_under_100ms(seeded_provider):
    """P50 of StubMemoryProvider.search() must be < 100ms."""
    samples = _measure(
        lambda: seeded_provider.search("python memory agent", SearchFilters(limit=5))
    )
    p50 = _p50(samples)
    print(f"\n[search] P50={p50:.1f}ms  P95={_p95(samples):.1f}ms  (n={len(samples)})")
    assert p50 < _P50_TARGET_MS, f"P50 {p50:.1f}ms exceeds {_P50_TARGET_MS}ms target"


# ---------------------------------------------------------------------------
# Benchmark: orchestrator turn overhead (excl. LLM)
# ---------------------------------------------------------------------------


def test_orchestrator_turn_overhead_p50_under_200ms(seeded_provider, tmp_path):
    """
    P50 of orchestrator._process_turn() with agent() mocked must be < 200ms.
    """
    from src.config import Config
    from src.hooks.post_session import PostSessionHook
    from src.orchestrator import Orchestrator

    mock_agent = MagicMock(return_value="mock response")

    cfg = MagicMock(spec=Config)
    cfg.session_inactivity_timeout_seconds = 300
    cfg.agent_turn_timeout_seconds = 30
    cfg.max_context_chars = 8000
    cfg.max_tool_calls = 10
    cfg.llm_max_tokens = 1024
    cfg.memory_root = str(tmp_path / "memory")

    post_session = MagicMock(spec=PostSessionHook)

    orch = Orchestrator(
        agent=mock_agent,
        memory_provider=seeded_provider,
        post_session_hook=post_session,
        config=cfg,
        session_id="bench_session",
    )

    samples = _measure(lambda: orch._process_turn("what do you know about python?"))
    p50 = _p50(samples)
    p95 = _p95(samples)
    print(f"\n[turn_overhead] P50={p50:.1f}ms  P95={p95:.1f}ms  (n={len(samples)})")
    assert p50 < 200, f"P50 {p50:.1f}ms exceeds 200ms target"


def test_orchestrator_turn_overhead_p95_under_200ms(seeded_provider, tmp_path):
    """P95 of orchestrator._process_turn() must be < 200ms."""
    from src.config import Config
    from src.hooks.post_session import PostSessionHook
    from src.orchestrator import Orchestrator

    mock_agent = MagicMock(return_value="mock response")

    cfg = MagicMock(spec=Config)
    cfg.session_inactivity_timeout_seconds = 300
    cfg.agent_turn_timeout_seconds = 30
    cfg.max_context_chars = 8000
    cfg.max_tool_calls = 10
    cfg.llm_max_tokens = 1024
    cfg.memory_root = str(tmp_path / "memory")

    orch = Orchestrator(
        agent=mock_agent,
        memory_provider=seeded_provider,
        post_session_hook=MagicMock(spec=PostSessionHook),
        config=cfg,
        session_id="bench_session_2",
    )

    samples = _measure(lambda: orch._process_turn("how do I deploy to production?"))
    p95 = _p95(samples)
    print(f"\n[turn_overhead] P95={p95:.1f}ms  (n={len(samples)})")
    assert p95 < 200, f"P95 {p95:.1f}ms exceeds 200ms target"
