"""
Per-loop state container attached to Strands ``Agent`` instances.

The Orchestrator (and the sub-agent wrapper) constructs one ``AgentStateContext``
per Agent, attaches it as ``agent.state_context``, and resets it at the
beginning of every agent loop.  The ``enforce_tool_budget`` hook in
``src/hooks/tool_budget.py`` reads this context on ``BeforeToolCallEvent`` to
decide whether to cancel a tool call.

Keeping the state on the Agent (instead of a threading.local or a hook
instance) means sub-agents naturally get their own isolated counters: the
``BeforeToolCallEvent`` carries ``event.agent``, so the hook just reads the
context off whichever agent is running — main agent or research sub-agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AgentStateContext:
    """
    Mutable per-loop state for an agent.

    The ``max_*`` / ``per_tool_limits`` fields configure budgets for the
    lifetime of the agent.  The counters (``tool_call_count``,
    ``per_tool_counts``, ``tools_invoked``) and loop metadata
    (``loop_started_at``, ``trace_id``) are reset at each new loop via
    :py:meth:`reset`.
    """

    max_tool_calls: int = 0
    per_tool_limits: Dict[str, int] = field(default_factory=dict)

    tool_call_count: int = 0
    per_tool_counts: Dict[str, int] = field(default_factory=dict)
    tools_invoked: List[str] = field(default_factory=list)

    loop_started_at: float = 0.0
    trace_id: str = ""

    def reset(self, trace_id: str = "") -> None:
        """Clear per-loop counters; keep configured limits.  Called per turn."""
        self.tool_call_count = 0
        self.per_tool_counts = {}
        self.tools_invoked = []
        self.loop_started_at = time.monotonic()
        self.trace_id = trace_id

    def check_and_increment(self, tool_name: str) -> Optional[str]:
        """
        Record one tool call and return a cancel-reason string when a budget
        is exceeded, else ``None``.

        The ordering matters: the global cap is checked first so an agent that
        has already hit the ceiling stops immediately regardless of which tool
        it is about to call next.  Tools not present in ``per_tool_limits`` are
        unrestricted beyond the global cap.
        """
        self.tool_call_count += 1
        if self.max_tool_calls and self.tool_call_count > self.max_tool_calls:
            return (
                f"tool call limit reached "
                f"({self.tool_call_count}/{self.max_tool_calls})"
            )

        if tool_name and tool_name in self.per_tool_limits:
            self.per_tool_counts[tool_name] = (
                self.per_tool_counts.get(tool_name, 0) + 1
            )
            cap = self.per_tool_limits[tool_name]
            if cap and self.per_tool_counts[tool_name] > cap:
                return (
                    f"per-tool budget reached for {tool_name} "
                    f"({self.per_tool_counts[tool_name]}/{cap})"
                )

        return None

    def unique_tools_invoked(self) -> List[str]:
        """Return ``tools_invoked`` deduped, preserving first-seen order."""
        seen: set[str] = set()
        out: List[str] = []
        for name in self.tools_invoked:
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out
