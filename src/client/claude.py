import anthropic
from .base import LLMClient
from typing import Any, Dict, List, Optional


class ClaudeClient(LLMClient):
    """
    Claude client implementation.
    """

    def setup_client(self):
        self.client = anthropic.Anthropic(api_key=self.token, base_url=self.api_endpoint)

    def completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-5-sonnet-20240620",
        max_tokens: int = 1024,
        **kwargs
    ) -> Any:
        """
        Generate completion using Anthropic API.
        """
        return self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs
        )

    def function_call(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        model: str = "claude-3-5-sonnet-20240620",
        max_tokens: int = 1024,
        **kwargs
    ) -> Any:
        """
        Perform function calling using Anthropic API.
        """
        return self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
            **kwargs
        )
