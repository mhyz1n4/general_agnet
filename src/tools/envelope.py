"""
Common result envelope for agent tools (V1.1 M1).

Every tool registered with the Strands agent returns the same shape so the
model and downstream orchestration code never have to branch on tool
identity to read success, payload, or error.

Shape
-----
``{"ok": bool, "data": Any, "error": str | None, "metadata": dict[str, Any]}``

``ok``       — True on success, False on failure.
``data``     — Payload on success; None on failure.
``error``    — Human-readable message on failure; None on success.
``metadata`` — Free-form supplementary information (timings, counts, source
               URLs, cache-hit flag, etc.). Always present; empty on no-op.

Helpers
-------
``ok(data, **metadata)``  — construct a success envelope.
``err(message, **metadata)`` — construct a failure envelope.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from typing_extensions import TypedDict


class ToolResult(TypedDict):
    """Typed result envelope every agent tool must return."""

    ok: bool
    data: Optional[Any]
    error: Optional[str]
    metadata: Dict[str, Any]


def ok(data: Any = None, **metadata: Any) -> ToolResult:
    """Construct a success ``ToolResult`` with ``data`` and optional metadata."""
    return ToolResult(ok=True, data=data, error=None, metadata=dict(metadata))


def err(message: str, **metadata: Any) -> ToolResult:
    """Construct a failure ``ToolResult`` with the error message and optional metadata."""
    return ToolResult(ok=False, data=None, error=message, metadata=dict(metadata))
