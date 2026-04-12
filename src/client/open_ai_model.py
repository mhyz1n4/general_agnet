import openai
from openai.types.chat import ChatCompletion, ChatCompletionToolParam
from typing import Dict, List

from .base import LLMClient


class OpenAIModelClient(LLMClient):
    """OpenAI-compatible client for open-source models or alternative providers."""

    def setup_client(self) -> None:
        """Instantiate the OpenAI-compatible client pointing at the configured base URL."""
        self.client = openai.OpenAI(api_key=self.token, base_url=self.api_endpoint)

    def completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs,
    ) -> ChatCompletion:
        """Generate a completion using an OpenAI-compatible Chat Completions API."""
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )

    def function_call(
        self,
        messages: List[Dict[str, str]],
        tools: List[ChatCompletionToolParam],
        model: str,
        **kwargs,
    ) -> ChatCompletion:
        """Perform tool/function calling using an OpenAI-compatible API."""
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs,
        )
