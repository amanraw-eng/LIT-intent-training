from __future__ import annotations

from .base import LLMClient
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient
from .. import config

_PROVIDERS: dict[str, type[LLMClient]] = {
    "gemini": GeminiClient,
    "openai": OpenAIClient,
}


def build_llm_client(provider: str | None = None, **kwargs) -> LLMClient:
    """Build an LLM client. `provider` defaults to config.DEFAULT_LLM_PROVIDER.
    Extra kwargs are forwarded to the client's constructor (e.g. model=...)."""
    provider = provider or config.DEFAULT_LLM_PROVIDER
    try:
        cls = _PROVIDERS[provider]
    except KeyError:
        raise ValueError(f"Unknown LLM provider {provider!r}. Available: {list(_PROVIDERS)}") from None
    return cls(**kwargs)
