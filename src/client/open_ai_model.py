import openai
from .base import LLMClient
from typing import Any, Dict, List, Optional


class OpenAIModelClient(LLMClient):
    """
    OpenAI compatible client implementation (for open source models or other providers).
    """

    def setup_client(self):
        self.client = openai.OpenAI(api_key=self.token, base_url=self.api_endpoint)

    def completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Any:
        """
        Generate completion using OpenAI compatible API.
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )

    def function_call(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], model: str, **kwargs) -> Any:
        """
        Perform function calling using OpenAI compatible API.
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs
        )
