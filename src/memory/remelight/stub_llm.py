"""
No-op LLM backend for ReMeLight.

Compactor and Summarizer require an OpenAI-compatible chat endpoint. V1.1 ships
without a real LLM wired in — semantic search and session persistence work
regardless, and memory compaction/summarization degrade to empty-string output
rather than crashing startup. Registering a concrete LLM is a V1.1 follow-up.
"""

from __future__ import annotations

from typing import Any

from agentscope.message import TextBlock
from agentscope.model import ChatModelBase
from agentscope.model._model_response import ChatResponse

from reme.core.registry_factory import R


STUB_LLM_BACKEND_NAME = "stub"


class _StubChatModel(ChatModelBase):
    """Returns an empty text block for every call."""

    def __init__(
        self,
        model_name: str = "stub",
        stream: bool = False,
        **_kwargs: Any,
    ) -> None:
        """
        Initialise the stub with a nominal model name.

        Streaming is forced off regardless of the ``stream`` argument so
        callers receive a single deterministic response. Extra kwargs are
        accepted and ignored to stay compatible with arbitrary ReMe configs.

        Args:
            model_name: Nominal model identifier surfaced in logs.
            stream:     Accepted for API parity; ignored (always ``False``).
            **_kwargs:  Accepted and discarded for config compatibility.
        """
        super().__init__(model_name=model_name, stream=False)

    async def __call__(self, *_args: Any, **_kwargs: Any) -> ChatResponse:
        """
        Return a ``ChatResponse`` containing a single empty text block.

        Args and kwargs are accepted and ignored so the stub slots into
        ReMe's dispatch without caring about message shape.

        Returns:
            ``ChatResponse`` with one empty ``TextBlock``.
        """
        return ChatResponse(content=[TextBlock(type="text", text="")])


def register_stub_llm() -> None:
    """Register the ``stub`` backend with ReMe's LLM registry (idempotent)."""
    if STUB_LLM_BACKEND_NAME not in R.as_llms:
        R.as_llms.register(STUB_LLM_BACKEND_NAME)(_StubChatModel)
