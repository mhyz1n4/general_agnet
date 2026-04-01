from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar


T = TypeVar("T", bound="LLMClient")


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    Implements a singleton pattern.
    """

    _instances: Dict[Type["LLMClient"], "LLMClient"] = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]

    def __init__(self, api_endpoint: str, token: str):
        """
        Initialize the LLM client with API endpoint and token.
        Only runs once per instance due to the singleton pattern.
        """
        if getattr(self, "_initialized", False):
            return
        self.api_endpoint = api_endpoint
        self.token = token
        self._initialized = True
        self.setup_client()

    @abstractmethod
    def setup_client(self):
        """
        Setup the internal client (e.g., OpenAI, Anthropic).
        """
        pass

    @abstractmethod
    def completion(self, *args, **kwargs) -> Any:
        """
        Generate a completion for a given prompt.
        """
        pass

    @abstractmethod
    def function_call(self, *args, **kwargs) -> Any:
        """
        Execute a function call with the LLM.
        """
        pass
