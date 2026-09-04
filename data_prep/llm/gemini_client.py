from __future__ import annotations

import json

from .base import LLMClient, LLMError, SchemaT
from .. import config


class GeminiClient(LLMClient):
    """Gemini on Vertex AI. Supports both text-only and multimodal (audio)
    structured output."""

    name = "gemini"

    def __init__(
        self,
        *,
        service_account_path: str | None = None,
        project: str | None = None,
        location: str | None = None,
        model: str | None = None,
    ):
        from google import genai
        from google.oauth2 import service_account

        service_account_path = service_account_path or config.GEMINI_KEY_PATH
        if not service_account_path:
            raise LLMError(
                "Gemini service account path not configured - set GEMINI_KEY_PATH in .env"
            )

        with open(service_account_path, encoding="utf-8") as f:
            info = json.load(f)

        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        self.client = genai.Client(
            vertexai=True,
            project=project or info.get("project_id") or config.GEMINI_PROJECT,
            location=location or config.GEMINI_LOCATION,
            credentials=credentials,
        )
        self.model = model or config.GEMINI_MODEL

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[SchemaT],
        *,
        temperature: float = 0.0,
        max_output_tokens: int | None = 8192,
    ) -> SchemaT:
        from google.genai import types

        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=f"{system_prompt}\n\n{user_prompt}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
        except Exception as e:
            raise LLMError(f"gemini call failed: {e}") from e
        if resp.parsed is None:
            raise LLMError(f"gemini returned unparseable response: {resp.text!r}")
        return resp.parsed

    def generate_structured_multimodal(
        self,
        system_prompt: str,
        text_prompt: str,
        audio_parts: list[tuple[bytes, str]],
        response_schema: type[SchemaT],
        *,
        temperature: float = 0.0,
        max_output_tokens: int | None = 8192,
    ) -> SchemaT:
        from google.genai import types

        parts = [types.Part.from_text(text=text_prompt)]
        for audio_bytes, mime_type in audio_parts:
            parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=mime_type))

        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    response_mime_type="application/json",
                    response_schema=response_schema.model_json_schema(),
                ),
            )
        except Exception as e:
            raise LLMError(f"gemini multimodal call failed: {e}") from e
        if not resp.text:
            raise LLMError("gemini multimodal response is empty")
        try:
            return response_schema.model_validate_json(resp.text)
        except Exception as e:
            raise LLMError(f"gemini multimodal response could not be parsed: {e}") from e
