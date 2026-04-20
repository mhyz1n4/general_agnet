"""
Strands hook that enforces per-agent tool-call budgets.

The callback is stateless: it reads the ``AgentStateContext`` attached to
the event's agent (``event.agent.state_context``) and asks the context to
update its counters.  When the context returns a cancel-reason, the hook
populates ``event.cancel_tool`` and Strands produces an error tool-result
for that call — the LLM sees the rejection and can decide how to proceed,
rather than the whole agent loop crashing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent

from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.agents.context import AgentStateContext

logger = get_logger(__name__)


def enforce_tool_budget(event: BeforeToolCallEvent) -> None:
    """
    Cancel the pending tool call when the agent's budget is exhausted.

    Reads the ``AgentStateContext`` off ``event.agent`` (set by whoever
    built the agent).  Agents without a ``state_context`` attribute are
    unrestricted — this keeps tests and experimental code trivial.
    """
    ctx: Optional["AgentStateContext"] = getattr(
        event.agent, "state_context", None
    )
    if ctx is None:
        return

    name = event.tool_use.get("name") or ""
    reason = ctx.check_and_increment(name)
    if reason is not None:
        logger.warning(
            "tool_budget: cancelling tool call",
            extra={"data": {"tool": name, "reason": reason}},
        )
        event.cancel_tool = reason
        return

    if name:
        ctx.tools_invoked.append(name)


class ToolBudgetHookProvider:
    """
    ``HookProvider`` adapter so ``Agent(hooks=[ToolBudgetHookProvider()])``
    wires ``enforce_tool_budget`` into the agent's registry.
    """

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        """Register ``enforce_tool_budget`` for ``BeforeToolCallEvent``."""
        registry.add_callback(BeforeToolCallEvent, enforce_tool_budget)
