"""
Unit tests for PreSessionHook and PostSessionHook.

V1.2: PreSessionHook creates ReMeLight directory structure (memory/, MEMORY.md,
memory.md).  PostSessionHook only writes metrics.
"""

import json
import os
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from src.hooks.pre_session import PreSessionHook
from src.hooks.post_session import PostSessionHook


# ===========================================================================
# PreSessionHook
# ===========================================================================


class TestPreSessionHook:
    """PreSessionHook validates system prerequisites before a session starts."""

    def test_all_healthy_returns_success(self, tmp_path: Path) -> None:
        """When FS is writable and no clients are configured, the hook must succeed."""
        hook: PreSessionHook = PreSessionHook(memory_root=str(tmp_path))
        result = hook.run()
        assert result.success is True
        assert result.degraded is False

    def test_fs_missing_creates_dir(self, tmp_path: Path) -> None:
        """A missing memory_root directory should be created automatically."""
        new_dir: str = str(tmp_path / "new_memory")
        hook: PreSessionHook = PreSessionHook(memory_root=new_dir)
        result = hook.run()
        assert result.success is True
        assert os.path.isdir(new_dir)

    def test_llm_auth_error_returns_failure(self, tmp_path: Path) -> None:
        """An LLM AuthenticationError must abort the session (success=False)."""
        class AuthenticationError(Exception):
            pass

        llm: MagicMock = MagicMock()
        llm.completion.side_effect = AuthenticationError("bad key")
        hook: PreSessionHook = PreSessionHook(memory_root=str(tmp_path), llm_client=llm)
        result = hook.run()
        assert result.success is False

    def test_llm_rate_limit_degrades(self, tmp_path: Path) -> None:
        """A RateLimitError from the LLM should degrade the session, not abort it."""
        class RateLimitError(Exception):
            pass

        llm: MagicMock = MagicMock()
        llm.completion.side_effect = RateLimitError("too many requests")
        hook: PreSessionHook = PreSessionHook(memory_root=str(tmp_path), llm_client=llm)
        result = hook.run()
        assert result.success is True
        assert result.degraded is True

    def test_no_redis_param_accepted(self, tmp_path: Path) -> None:
        """PreSessionHook no longer accepts redis_client — verify it works without."""
        hook: PreSessionHook = PreSessionHook(memory_root=str(tmp_path))
        result = hook.run()
        assert result.success is True

    def test_creates_remelight_directories(self, tmp_path: Path) -> None:
        """PreSessionHook must create memory/ subdir and index files."""
        root: str = str(tmp_path / "fresh_memory")
        hook: PreSessionHook = PreSessionHook(memory_root=root)
        result = hook.run()
        assert result.success is True
        assert os.path.isdir(os.path.join(root, "memory"))
        assert os.path.isfile(os.path.join(root, "MEMORY.md"))
        assert os.path.isfile(os.path.join(root, "memory.md"))

    def test_existing_index_files_not_overwritten(self, tmp_path: Path) -> None:
        """If MEMORY.md already has content, it must not be truncated."""
        root: str = str(tmp_path)
        memory_md = os.path.join(root, "MEMORY.md")
        with open(memory_md, "w") as f:
            f.write("# Existing content\n")
        hook: PreSessionHook = PreSessionHook(memory_root=root)
        hook.run()
        with open(memory_md) as f:
            assert f.read() == "# Existing content\n"


# ===========================================================================
# PostSessionHook
# ===========================================================================


class TestPostSessionHook:
    """PostSessionHook writes session metrics to JSONL."""

    def _make_hook(self, tmp_path: Path) -> PostSessionHook:
        """Build a PostSessionHook with a temporary metrics path."""
        return PostSessionHook(
            session_id="test-session",
            metrics_path=str(tmp_path / "metrics.jsonl"),
        )

    def test_run_returns_success(self, tmp_path: Path) -> None:
        """run() must return success=True immediately (work is done in a thread)."""
        hook = self._make_hook(tmp_path)
        result = hook.run(metrics={})
        assert result.success is True

    def test_metrics_written_after_run(self, tmp_path: Path) -> None:
        """_run_async() must write a JSONL line to the metrics file."""
        hook = self._make_hook(tmp_path)
        hook._run_async({"turn_count": 3})
        path: str = str(tmp_path / "metrics.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            record = json.loads(f.readline())
        assert record["session_id"] == "test-session"

    def test_zero_turns_no_crash(self, tmp_path: Path) -> None:
        """An empty turns list should not raise any exception."""
        hook = self._make_hook(tmp_path)
        hook._run_async({"turns": []})

    def test_metrics_excludes_turns(self, tmp_path: Path) -> None:
        """The persisted metrics record should exclude the turns list for compactness."""
        hook = self._make_hook(tmp_path)
        hook._run_async({"turn_count": 1, "turns": [{"user": "hi", "assistant": "hello"}]})
        path: str = str(tmp_path / "metrics.jsonl")
        with open(path) as f:
            record = json.loads(f.readline())
        assert "turns" not in record
