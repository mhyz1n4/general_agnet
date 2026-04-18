"""
Risky-action confirmation hook (V1.1 M4).

Tools that mutate the user's system (file writes, code execution, outbound
network calls to non-search endpoints, etc.) should wrap themselves in
``@requires_confirmation("short description")``.  At invocation time the
decorator consults ``confirm_or_deny`` which either prompts on TTY stdin or
returns ``False`` when no user is attached.

Mode resolution
---------------
1. Per-call override passed to ``confirm_or_deny(..., mode=...)``.
2. Process-wide override set via ``set_confirmation_mode`` (a ContextVar so
   tests can scope it).
3. Auto-detect: ``sys.stdin.isatty()`` → ``interactive`` else ``non_interactive``.

``non_interactive`` is always deny-by-default: if no one is there to say yes,
we refuse. This is the security invariant.
"""

from __future__ import annotations

import contextvars
import functools
import sys
from typing import Callable, Literal, Optional, TypeVar

from src.logging_config import get_logger
from src.tools.envelope import ToolResult, err

logger = get_logger(__name__)

Mode = Literal["interactive", "non_interactive", "auto"]

_mode_override: contextvars.ContextVar[Optional[Mode]] = contextvars.ContextVar(
    "confirmation_mode_override", default=None
)

_YES_ANSWERS = frozenset({"y", "yes"})


def set_confirmation_mode(mode: Optional[Mode]) -> contextvars.Token:
    """
    Set the process-wide (ContextVar-scoped) confirmation mode.

    Returns the token so callers can restore the previous value with
    ``reset_confirmation_mode``. ``None`` means "fall back to auto-detect".
    """
    return _mode_override.set(mode)


def reset_confirmation_mode(token: contextvars.Token) -> None:
    """Restore the confirmation mode override to its previous value."""
    _mode_override.reset(token)


def _resolve_mode(explicit: Optional[Mode]) -> Mode:
    """Pick the effective mode given an optional explicit override."""
    if explicit is not None and explicit != "auto":
        return explicit
    override = _mode_override.get()
    if override is not None and override != "auto":
        return override
    try:
        return "interactive" if sys.stdin.isatty() else "non_interactive"
    except (ValueError, OSError):
        return "non_interactive"


def confirm_or_deny(
    action_description: str,
    mode: Optional[Mode] = None,
    input_fn: Callable[[str], str] = input,
) -> bool:
    """
    Ask the user whether to proceed with ``action_description``.

    Args:
        action_description: One-line summary of what the tool is about to do.
        mode:               Force a specific mode; ``None`` defers to the
                            ContextVar override or stdin auto-detect.
        input_fn:           Callable used to read the user's answer
                            (dependency-injected so tests can provide a stub).

    Returns:
        ``True`` if the user affirmed (interactive mode only) or
        ``False`` if they declined / we are in non-interactive mode.
    """
    effective = _resolve_mode(mode)
    if effective == "non_interactive":
        logger.warning(
            "confirm_or_deny: auto-denied (non-interactive mode)",
            extra={"data": {"action": action_description}},
        )
        return False

    prompt = f"[confirm] {action_description} [y/N]: "
    try:
        answer = input_fn(prompt)
    except (EOFError, KeyboardInterrupt):
        logger.warning(
            "confirm_or_deny: denied (stdin closed or interrupted)",
            extra={"data": {"action": action_description}},
        )
        return False
    return answer.strip().casefold() in _YES_ANSWERS


F = TypeVar("F", bound=Callable[..., ToolResult])


def requires_confirmation(description: str) -> Callable[[F], F]:
    """
    Decorate a tool function so it asks for confirmation before executing.

    The wrapped function is expected to return a ``ToolResult``; on denial the
    decorator returns a failure envelope without invoking the wrapped body.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> ToolResult:
            if not confirm_or_deny(description):
                logger.info(
                    "requires_confirmation: action denied",
                    extra={"data": {"tool": fn.__name__, "action": description}},
                )
                return err(
                    f"action denied: {description}",
                    tool=fn.__name__,
                )
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
