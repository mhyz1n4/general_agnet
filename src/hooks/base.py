"""Base classes for the hook system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HookResult:
    """Result returned by every hook."""
    success: bool
    message: str
    degraded: bool = False
    data: Any = field(default=None)


class BaseHook(ABC):
    """Abstract base for all hooks."""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> HookResult:
        """Execute the hook and return a result."""
