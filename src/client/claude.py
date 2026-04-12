import anthropic
from anthropic.types import Message, ToolParam
from typing import Dict, List

from .base import LLMClient
from src.constants import DEFAULT_CLAUDE_MODEL, DEFAULT_LLM_MAX_TOKENS


class ClaudeClient(LLMClient):
    """Claude client implementation using the Anthropic SDK."""

    def setup_client(self) -> None:
        """Instantiate the Anthropic SDK client using the configured API key and endpoint."""
        self.client = anthropic.Anthropic(api_key=self.token, base_url=self.api_endpoint)

    def completion(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_CLAUDE_MODEL,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        **kwargs,
    ) -> Message:
        """Generate a completion using the Anthropic Messages API."""
        return self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs,
        )

    def function_call(
        self,
        messages: List[Dict[str, str]],
        tools: List[ToolParam],
        model: str = DEFAULT_CLAUDE_MODEL,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        **kwargs,
    ) -> Message:
        """Perform tool/function calling using the Anthropic Messages API."""
        return self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
            **kwargs,
        )
