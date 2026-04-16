from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, TypeVar


T = TypeVar("T", bound="LLMClient")


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    Implements a singleton pattern.
    """

    _instances: Dict[Type["LLMClient"], "LLMClient"] = {}

    def __new__(cls, *args, **kwargs):
        """
        Return the existing singleton instance for *cls*, creating it if needed.

        The instance is stored in ``_instances`` keyed by concrete class so
        that each subclass maintains its own independent singleton.
        """
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
    def completion(self, *args, **kwargs) -> object:
        """
        Generate a completion for a given prompt.
        Concrete subclasses narrow the return type to the provider-specific
        response object (e.g. anthropic.types.Message, openai ChatCompletion).
        """

    @abstractmethod
    def function_call(self, *args, **kwargs) -> object:
        """
        Execute a function call with the LLM.
        Concrete subclasses narrow the return type to the provider-specific
        response object.
        """
