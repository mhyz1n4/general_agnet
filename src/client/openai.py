import openai
from openai.types.chat import ChatCompletion, ChatCompletionToolParam
from typing import Dict, List

from .base import LLMClient
from src.constants import DEFAULT_OPENAI_MODEL


class OpenAIClient(LLMClient):
    """OpenAI client implementation."""

    def setup_client(self) -> None:
        """Instantiate the OpenAI SDK client using the configured API key and endpoint."""
        self.client = openai.OpenAI(api_key=self.token, base_url=self.api_endpoint)

    def completion(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_OPENAI_MODEL,
        **kwargs,
    ) -> ChatCompletion:
        """Generate a completion using the OpenAI Chat Completions API."""
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )

    def function_call(
        self,
        messages: List[Dict[str, str]],
        tools: List[ChatCompletionToolParam],
        model: str = DEFAULT_OPENAI_MODEL,
        **kwargs,
    ) -> ChatCompletion:
        """Perform tool/function calling using the OpenAI Chat Completions API."""
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs,
        )
