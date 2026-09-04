"""Provider-agnostic structured-output LLM client interface.

Concrete clients (gemini_client.py, openai_client.py) only implement the
actual provider call. Batching, concurrency, and retries are handled
uniformly by data_prep.llm.concurrency for every provider and every task
(generation, text relabeling, multimodal relabeling) - clients never
retry or batch on their own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMError(Exception):
    """Raised for any provider call failure or unparseable structured response."""


class LLMClient(ABC):
    """Common interface for structured-output LLM calls."""

    name: str
    model: str

    @abstractmethod
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[SchemaT],
        *,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> SchemaT:
        """Run one text-only structured-output call, returning the parsed model."""

    def generate_structured_multimodal(
        self,
        system_prompt: str,
        text_prompt: str,
        audio_parts: list[tuple[bytes, str]],
        response_schema: type[SchemaT],
        *,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> SchemaT:
        """Run one structured-output call with audio input alongside text.

        `audio_parts` is a list of (audio_bytes, mime_type) tuples, one per
        item in the batch, in order. Not every provider supports this.
        """
        raise NotImplementedError(f"{self.name} client does not support multimodal (audio) input")
