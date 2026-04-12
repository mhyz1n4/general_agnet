"""Base types for the sub-agent system."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SubAgentResult:
    """
    Typed result returned by every sub-agent.

    Sub-agents are Python callables invoked synchronously by the orchestrator.
    The orchestrator checks ``success`` before injecting ``output`` into the next
    prompt; on ``success=False`` it follows the error-recovery policy (§5c).

    Attributes:
        success: True if the agent completed its task without errors.
        output:  Human-readable result string (empty string on failure).
        error:   Error description when ``success`` is False, otherwise None.
    """

    success: bool
    output: str
    error: Optional[str] = None
