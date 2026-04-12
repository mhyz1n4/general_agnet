"""Base classes for the hook system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class HookResult:
    """
    Result returned by every hook.

    Attributes:
        success:  ``True`` if the hook completed without a fatal error.
        message:  Human-readable outcome or processed value (e.g. normalised
                  query, truncated context).  Empty string when not applicable.
        degraded: ``True`` when the system can continue in a reduced-capability
                  mode (e.g. Redis unavailable).  Always ``False`` when
                  ``success`` is ``False``.
        data:     Optional structured payload for callers that need more than
                  the message string.
    """

    success: bool
    message: str
    degraded: bool = False
    data: Optional[object] = field(default=None)


class BaseHook(ABC):
    """Abstract base for all hooks."""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> HookResult:
        """Execute the hook and return a result."""
