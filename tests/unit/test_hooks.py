"""Unit tests for all four hooks (all deps mocked)."""

import pytest
from unittest.mock import MagicMock, patch

from src.hooks.pre_mem_fetch import PreMemFetchHook
from src.hooks.post_mem_fetch import PostMemFetchHook
from src.hooks.pre_session import PreSessionHook
from src.hooks.post_session import PostSessionHook


# ===========================================================================
# PreMemFetchHook
# ===========================================================================

class TestPreMemFetchHook:
    @pytest.fixture(autouse=True)
    def hook(self):
        self.hook = PreMemFetchHook()

    def test_empty_string(self):
        assert self.hook.run("").message == ""

    def test_special_chars_stripped(self):
        result = self.hook.run("!@#$%")
        assert result.message == ""

    def test_apostrophe_kept(self):
        result = self.hook.run("what's up")
        assert result.message == "what's up"

    def test_lowercased(self):
        result = self.hook.run("Hello World")
        assert result.message == "hello world"

    def test_collapse_whitespace(self):
        result = self.hook.run("hello  world")
        assert result.message == "hello world"

    def test_strip_leading_trailing(self):
        result = self.hook.run("  hello ")
        assert result.message == "hello"

    def test_long_query_processed(self):
        query = "a " * 2500
        result = self.hook.run(query)
        assert result.success is True
        assert len(result.message) > 0

    def test_success_true(self):
        assert self.hook.run("test").success is True


# ===========================================================================
# PostMemFetchHook
# ===========================================================================

class TestPostMemFetchHook:
    def test_context_under_limit_unchanged(self):
        hook = PostMemFetchHook(max_context_chars=1000)
        ctx = "hello"
        result = hook.run(context=ctx)
        assert result.message == ctx

    def test_empty_context_unchanged(self):
        hook = PostMemFetchHook(max_context_chars=100)
        result = hook.run(context="")
        assert result.message == ""

    def test_over_limit_truncates_blocks(self):
        block = "--- Context (Key: k1) ---\n" + "x" * 200 + "\n"
        ctx = block * 5  # 5 identical blocks
        hook = PostMemFetchHook(max_context_chars=len(block) * 2 + 10)
        result = hook.run(context=ctx)
        assert len(result.message) <= len(block) * 2 + 20

    def test_single_oversized_block_returned_as_is(self):
        block = "--- Context (Key: k1) ---\n" + "x" * 5000 + "\n"
        hook = PostMemFetchHook(max_context_chars=100)
        result = hook.run(context=block)
        assert result.message == block

    def test_saves_retrieval_event_to_session_storage(self):
        session_storage = MagicMock()
        hook = PostMemFetchHook(max_context_chars=8000, session_storage=session_storage)
        hook.run(context="some context", retrieved_keys=["k1", "k2"], query="test query")
        session_storage.save.assert_called_once()

    def test_empty_keys_no_crash(self):
        hook = PostMemFetchHook()
        result = hook.run(context="", retrieved_keys=[])
        assert result.success is True


# ===========================================================================
# PreSessionHook
# ===========================================================================

class TestPreSessionHook:
    def test_all_healthy_returns_success(self, tmp_path):
        hook = PreSessionHook(memory_root=str(tmp_path))
        result = hook.run()
        assert result.success is True
        assert result.degraded is False

    def test_fs_missing_creates_dir(self, tmp_path):
        new_dir = str(tmp_path / "new_memory")
        hook = PreSessionHook(memory_root=new_dir)
        result = hook.run()
        assert result.success is True
        import os
        assert os.path.isdir(new_dir)

    def test_redis_connection_error_degrades(self, tmp_path):
        redis_client = MagicMock()
        redis_client.ping.side_effect = ConnectionError("no redis")
        hook = PreSessionHook(memory_root=str(tmp_path), redis_client=redis_client)
        result = hook.run()
        assert result.success is True
        assert result.degraded is True
        assert "Redis" in result.message

    def test_llm_auth_error_returns_failure(self, tmp_path):
        class AuthenticationError(Exception):
            pass

        llm = MagicMock()
        llm.completion.side_effect = AuthenticationError("bad key")
        hook = PreSessionHook(memory_root=str(tmp_path), llm_client=llm)
        result = hook.run()
        assert result.success is False

    def test_llm_rate_limit_degrades(self, tmp_path):
        class RateLimitError(Exception):
            pass

        llm = MagicMock()
        llm.completion.side_effect = RateLimitError("too many requests")
        hook = PreSessionHook(memory_root=str(tmp_path), llm_client=llm)
        result = hook.run()
        assert result.success is True
        assert result.degraded is True

    def test_redis_and_llm_rate_limit_both_degrade(self, tmp_path):
        class RateLimitError(Exception):
            pass

        redis_client = MagicMock()
        redis_client.ping.side_effect = ConnectionError("no redis")
        llm = MagicMock()
        llm.completion.side_effect = RateLimitError("rate limited")
        hook = PreSessionHook(
            memory_root=str(tmp_path), redis_client=redis_client, llm_client=llm
        )
        result = hook.run()
        assert result.success is True
        assert result.degraded is True


# ===========================================================================
# PostSessionHook
# ===========================================================================

class TestPostSessionHook:
    def _make_hook(self, tmp_path, redis_client=None, llm_client=None):
        mm = MagicMock()
        return PostSessionHook(
            memory_manager=mm,
            session_id="test-session",
            redis_client=redis_client,
            llm_client=llm_client,
            metrics_path=str(tmp_path / "metrics.jsonl"),
        ), mm

    def test_run_returns_success(self, tmp_path):
        hook, _ = self._make_hook(tmp_path)
        result = hook.run(metrics={})
        assert result.success is True

    def test_metrics_written_after_run(self, tmp_path):
        import time
        hook, _ = self._make_hook(tmp_path)
        hook.run(metrics={"turn_count": 3})
        # Give daemon thread a moment
        time.sleep(0.3)
        import os, json
        path = str(tmp_path / "metrics.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            record = json.loads(f.readline())
        assert record["session_id"] == "test-session"

    def test_zero_turns_no_crash(self, tmp_path):
        import time
        hook, _ = self._make_hook(tmp_path)
        hook.run(metrics={"turns": []})
        time.sleep(0.3)

    def test_no_redis_client_skips_flush(self, tmp_path):
        import time
        hook, mm = self._make_hook(tmp_path)
        hook.run(metrics={})
        time.sleep(0.2)
        mm.save_message.assert_not_called()  # only session summary would be saved
