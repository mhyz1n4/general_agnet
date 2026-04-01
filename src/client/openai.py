import openai
from .base import LLMClient
from typing import Any, Dict, List, Optional


class OpenAIClient(LLMClient):
    """
    OpenAI client implementation.
    """

    def setup_client(self):
        self.client = openai.OpenAI(api_key=self.token, base_url=self.api_endpoint)

    def completion(self, messages: List[Dict[str, str]], model: str = "gpt-4o", **kwargs) -> Any:
        """
        Generate completion using OpenAI API.
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )

    def function_call(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], model: str = "gpt-4o", **kwargs) -> Any:
        """
        Perform function calling using OpenAI API.
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs
        )
