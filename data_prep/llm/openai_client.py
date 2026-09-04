from __future__ import annotations

from .base import LLMClient, LLMError, SchemaT
from .. import config

# These model families reject a custom `temperature` value - only the
# provider default is accepted.
_NO_CUSTOM_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class OpenAIClient(LLMClient):
    """Text-only structured output via OpenAI's chat completions API."""

    name = "openai"

    def __init__(self, *, api_key: str | None = None, model: str | None = None):
        from openai import OpenAI

        api_key = api_key or config.OPENAI_API_KEY
        if not api_key:
            raise LLMError("OPENAI_API_KEY not set - add it to data_prep/.env or the environment")
        self.client = OpenAI(api_key=api_key)
        self.model = model or config.OPENAI_MODEL

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[SchemaT],
        *,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> SchemaT:
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_schema,
        )
        if not self.model.startswith(_NO_CUSTOM_TEMPERATURE_PREFIXES):
            kwargs["temperature"] = temperature
        try:
            resp = self.client.chat.completions.parse(**kwargs)
        except Exception as e:
            raise LLMError(f"openai call failed: {e}") from e
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise LLMError(f"openai returned unparseable response: {resp.choices[0].message.content!r}")
        return parsed
