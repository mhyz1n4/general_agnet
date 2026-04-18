"""
Unit tests for the confirmation hook (V1.1 M4).
"""

from __future__ import annotations

import pytest

from src.tools import confirm as confirm_mod
from src.tools.confirm import (
    confirm_or_deny,
    requires_confirmation,
    reset_confirmation_mode,
    set_confirmation_mode,
)
from src.tools.envelope import ok


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------


def test_explicit_non_interactive_denies_without_prompt() -> None:
    """``mode='non_interactive'`` returns False and never calls input_fn."""
    calls: list[str] = []

    def fake_input(prompt: str) -> str:
        calls.append(prompt)
        return "y"

    assert confirm_or_deny("delete prod", mode="non_interactive", input_fn=fake_input) is False
    assert calls == []


def test_interactive_yes_confirms() -> None:
    """A 'y' answer returns True in interactive mode."""
    assert confirm_or_deny("x", mode="interactive", input_fn=lambda _p: "y") is True


def test_interactive_whitespace_and_case() -> None:
    """Answers are stripped and case-folded before comparison."""
    assert confirm_or_deny("x", mode="interactive", input_fn=lambda _p: "  YES  ") is True


def test_interactive_other_answers_deny() -> None:
    """Anything other than y/yes is treated as a decline."""
    assert confirm_or_deny("x", mode="interactive", input_fn=lambda _p: "n") is False
    assert confirm_or_deny("x", mode="interactive", input_fn=lambda _p: "") is False
    assert confirm_or_deny("x", mode="interactive", input_fn=lambda _p: "maybe") is False


def test_interactive_eof_denies() -> None:
    """EOFError on stdin returns a deny without propagating."""
    def boom(_p: str) -> str:
        raise EOFError

    assert confirm_or_deny("x", mode="interactive", input_fn=boom) is False


def test_contextvar_override_wins_over_auto() -> None:
    """``set_confirmation_mode`` forces the mode even when stdin is a TTY."""
    token = set_confirmation_mode("non_interactive")
    try:
        assert confirm_or_deny("x", input_fn=lambda _p: "y") is False
    finally:
        reset_confirmation_mode(token)


def test_auto_mode_falls_back_to_non_interactive_when_stdin_not_tty(monkeypatch) -> None:
    """When stdin is not a TTY, auto resolves to non_interactive (deny)."""
    monkeypatch.setattr(confirm_mod.sys.stdin, "isatty", lambda: False, raising=False)
    assert confirm_or_deny("x", input_fn=lambda _p: "y") is False


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def test_decorator_runs_body_on_confirm(monkeypatch) -> None:
    """When confirmation returns True the wrapped tool runs and its envelope passes through."""
    token = set_confirmation_mode("interactive")
    try:
        @requires_confirmation("execute X")
        def stub_tool() -> dict:
            return ok({"ran": True})

        monkeypatch.setattr("builtins.input", lambda _p: "y")
        result = stub_tool()

        assert result["ok"] is True
        assert result["data"] == {"ran": True}
    finally:
        reset_confirmation_mode(token)


def test_decorator_returns_error_envelope_on_deny() -> None:
    """When confirmation denies, the wrapped body is skipped and err is returned."""
    token = set_confirmation_mode("non_interactive")
    try:
        called: list[int] = []

        @requires_confirmation("run python code")
        def stub_tool() -> dict:
            called.append(1)
            return ok({"ran": True})

        result = stub_tool()
        assert result["ok"] is False
        assert "run python code" in result["error"]
        assert result["metadata"]["tool"] == "stub_tool"
        assert called == []
    finally:
        reset_confirmation_mode(token)
