"""
Unit tests for the ``run_python`` tool (V1.1 M6).

``subprocess.run`` is stubbed in every test so the suite never actually spawns
a Python interpreter.  We assert on the ``ToolResult`` envelope, truncation,
timeout handling, the feature flag, and confirmation-hook wiring.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.tools import run_python as rp
from src.tools.confirm import reset_confirmation_mode, set_confirmation_mode


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


def test_factory_returns_none_when_disabled() -> None:
    """With ``enable=False`` the factory short-circuits and registers nothing."""
    assert rp.create_run_python_tool(enable=False) is None


def test_factory_from_config_respects_flag() -> None:
    """``create_run_python_tool_from_config`` honours ``enable_code_exec``."""
    cfg_off = SimpleNamespace(enable_code_exec=False)
    cfg_on = SimpleNamespace(enable_code_exec=True)
    assert rp.create_run_python_tool_from_config(cfg_off) is None
    # With the flag on a tool is returned.  We do not execute it here.
    tool = rp.create_run_python_tool_from_config(cfg_on)
    assert tool is not None


# ---------------------------------------------------------------------------
# Happy path & error branches (confirmation forced to interactive)
# ---------------------------------------------------------------------------


@pytest.fixture
def tool():
    """Build the tool with confirmation forced to auto-approve via monkeypatched input."""
    token = set_confirmation_mode("interactive")
    yield rp.create_run_python_tool(enable=True)
    reset_confirmation_mode(token)


@pytest.fixture(autouse=True)
def _auto_yes(monkeypatch):
    """Auto-answer 'y' to every confirmation prompt in this module."""
    monkeypatch.setattr("builtins.input", lambda _p: "y")


def _fake_completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess:
    """Minimal ``CompletedProcess`` stand-in matching what ``subprocess.run`` returns."""
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_empty_code_returns_err_without_running(tool) -> None:
    """Whitespace-only code is rejected before subprocess is invoked."""
    with patch.object(rp.subprocess, "run") as run:
        result = tool(code="   \n\t ")
    assert result["ok"] is False
    assert "empty" in result["error"]
    run.assert_not_called()


def test_happy_path_returns_ok_with_stdout(tool) -> None:
    """Exit 0 gives an ok envelope carrying stdout/stderr/returncode."""
    with patch.object(
        rp.subprocess, "run", return_value=_fake_completed(stdout="hi\n")
    ) as run:
        result = tool(code="print('hi')")

    assert result["ok"] is True
    assert result["data"]["stdout"] == "hi\n"
    assert result["data"]["returncode"] == 0
    assert result["data"]["truncated"] is False
    # ``-I`` isolates the child interpreter; verify it was passed.
    argv = run.call_args[0][0]
    assert "-I" in argv
    assert argv[-1] == "print('hi')"


def test_nonzero_exit_returns_err_with_data(tool) -> None:
    """A non-zero returncode is surfaced as err but still carries captured output."""
    with patch.object(
        rp.subprocess,
        "run",
        return_value=_fake_completed(stderr="boom\n", returncode=1),
    ):
        result = tool(code="raise SystemExit(1)")

    assert result["ok"] is False
    assert "exited 1" in result["error"]
    # Non-zero-exit payload rides in metadata since err() nulls the top-level data.
    payload = result["metadata"]["data"]
    assert payload["stderr"] == "boom\n"
    assert payload["returncode"] == 1


def test_timeout_returns_err_envelope(tool) -> None:
    """``TimeoutExpired`` is caught and reported without crashing."""
    with patch.object(
        rp.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(cmd="python", timeout=5),
    ):
        result = tool(code="while True: pass", timeout_s=5)

    assert result["ok"] is False
    assert "timed out" in result["error"]
    assert result["metadata"]["timeout_s"] == 5


def test_generic_subprocess_error_is_caught(tool) -> None:
    """Unexpected OSError from ``subprocess.run`` returns a clean err envelope."""
    with patch.object(
        rp.subprocess, "run", side_effect=OSError("fork failed")
    ):
        result = tool(code="print(1)")

    assert result["ok"] is False
    assert "fork failed" in result["error"]


def test_stdout_is_truncated_past_max_output_bytes() -> None:
    """Oversized stdout is clipped and the ``truncated`` flag flips."""
    token = set_confirmation_mode("interactive")
    try:
        tool = rp.create_run_python_tool(enable=True, max_output_bytes=16)
        big = "x" * 100
        with patch.object(
            rp.subprocess, "run", return_value=_fake_completed(stdout=big)
        ):
            result = tool(code="print('x' * 100)")
    finally:
        reset_confirmation_mode(token)

    assert result["ok"] is True
    assert len(result["data"]["stdout"]) == 16
    assert result["data"]["truncated"] is True


def test_timeout_is_capped_at_hard_limit(tool) -> None:
    """A model-supplied ``timeout_s`` above the cap is clamped to the cap."""
    captured = {}

    def _capture(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return _fake_completed()

    with patch.object(rp.subprocess, "run", side_effect=_capture):
        tool(code="print(1)", timeout_s=9999)

    assert captured["timeout"] == rp._DEFAULT_TIMEOUT_CAP


# ---------------------------------------------------------------------------
# Confirmation wiring
# ---------------------------------------------------------------------------


def test_non_interactive_mode_denies_without_invoking_subprocess() -> None:
    """When confirmation resolves to non_interactive the body never runs."""
    token = set_confirmation_mode("non_interactive")
    try:
        tool = rp.create_run_python_tool(enable=True)
        with patch.object(rp.subprocess, "run") as run:
            result = tool(code="print(1)")
    finally:
        reset_confirmation_mode(token)

    assert result["ok"] is False
    assert "action denied" in result["error"]
    run.assert_not_called()
