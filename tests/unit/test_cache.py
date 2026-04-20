"""
Unit tests for the per-session tool-result cache (V1.1 M7).
"""

from __future__ import annotations

import pytest

from src.tools.cache import (
    DEFAULT_MAX_ENTRIES,
    SessionToolCache,
    _hash_args,
    cached_tool,
    get_session_cache,
    reset_session_cache,
    set_session_cache,
)
from src.tools.envelope import err, ok


# ---------------------------------------------------------------------------
# SessionToolCache — eviction + hit/miss accounting
# ---------------------------------------------------------------------------


def test_miss_then_hit_accounting() -> None:
    """A ``get`` on an unknown key misses; a subsequent ``set``+``get`` hits."""
    cache = SessionToolCache()
    assert cache.get("tool", "k1") is None
    assert cache.misses == 1
    assert cache.hits == 0

    cache.set("tool", "k1", ok({"v": 1}))
    assert cache.get("tool", "k1")["data"] == {"v": 1}
    assert cache.hits == 1


def test_lru_eviction_drops_oldest() -> None:
    """When the cache overflows, the least-recently-used entry is evicted."""
    cache = SessionToolCache(max_entries=2)
    cache.set("t", "a", ok(1))
    cache.set("t", "b", ok(2))
    cache.get("t", "a")                 # touch "a" so "b" becomes oldest
    cache.set("t", "c", ok(3))          # should evict "b"

    assert cache.get("t", "a") is not None
    assert cache.get("t", "c") is not None
    assert cache.get("t", "b") is None
    assert len(cache) == 2


def test_repeat_set_moves_to_end_and_does_not_grow() -> None:
    """Re-setting the same key updates the value and keeps length stable."""
    cache = SessionToolCache(max_entries=2)
    cache.set("t", "a", ok(1))
    cache.set("t", "b", ok(2))
    cache.set("t", "a", ok(99))          # "b" is now oldest
    cache.set("t", "c", ok(3))           # evict "b"

    assert cache.get("t", "a")["data"] == 99
    assert cache.get("t", "b") is None
    assert cache.get("t", "c")["data"] == 3


# ---------------------------------------------------------------------------
# _hash_args — determinism & kwarg-order independence
# ---------------------------------------------------------------------------


def test_hash_args_is_deterministic() -> None:
    """Same inputs give the same hash across calls."""
    assert _hash_args(("x",), {"a": 1}) == _hash_args(("x",), {"a": 1})


def test_hash_args_ignores_kwarg_order() -> None:
    """Keyword argument order does not change the hash."""
    assert _hash_args((), {"a": 1, "b": 2}) == _hash_args((), {"b": 2, "a": 1})


def test_hash_args_changes_on_value_change() -> None:
    """Different call arguments produce different hashes."""
    assert _hash_args((), {"a": 1}) != _hash_args((), {"a": 2})


# ---------------------------------------------------------------------------
# @cached_tool decorator
# ---------------------------------------------------------------------------


@pytest.fixture
def bound_cache():
    """Provide a fresh cache bound to the current context."""
    cache = SessionToolCache()
    token = set_session_cache(cache)
    yield cache
    reset_session_cache(token)


def test_cached_tool_second_call_returns_cached_envelope(bound_cache) -> None:
    """Identical calls run the wrapped body exactly once; second call is a cache hit."""
    calls: list[int] = []

    @cached_tool("my_tool")
    def fn(x: int) -> dict:
        calls.append(x)
        return ok({"x": x})

    r1 = fn(x=1)
    r2 = fn(x=1)
    assert r1 is r2
    assert calls == [1]
    assert bound_cache.hits == 1


def test_cached_tool_differentiates_on_args(bound_cache) -> None:
    """Different arguments are cached separately."""
    @cached_tool("my_tool")
    def fn(x: int) -> dict:
        return ok({"x": x})

    fn(x=1)
    fn(x=2)
    fn(x=1)
    assert bound_cache.hits == 1
    assert bound_cache.misses == 2


def test_cached_tool_does_not_cache_failures(bound_cache) -> None:
    """``err()`` envelopes are not cached so transient failures can recover."""
    calls: list[int] = []

    @cached_tool("my_tool")
    def fn() -> dict:
        calls.append(1)
        return err("boom")

    fn()
    fn()
    assert calls == [1, 1]
    assert bound_cache.hits == 0


def test_cached_tool_noop_when_no_cache_bound() -> None:
    """With no cache on the context, the decorator is transparent."""
    calls: list[int] = []

    @cached_tool("my_tool")
    def fn() -> dict:
        calls.append(1)
        return ok({"v": 1})

    # No set_session_cache — default is None.
    fn()
    fn()
    assert calls == [1, 1]
    assert get_session_cache() is None


def test_default_max_entries_is_sane() -> None:
    """Default LRU size aligns with the plan (64)."""
    assert DEFAULT_MAX_ENTRIES == 64
