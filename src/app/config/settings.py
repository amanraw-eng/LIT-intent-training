"""Deployment-specific settings, loaded from the environment / .env.

Only real secrets and values that legitimately differ per deployment live
here (which HF model to serve, credentials, where to listen, log
verbosity). Fixed, code-level defaults live in constants.py instead.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    HF_MODEL_REPO: str = "amn-raw/whisper-small-intent17-classifier"
    HF_TOKEN: str = ""

    LOG_LEVEL: str = "INFO"
    # Set False in production if you don't want exception details exposed
    # in API error responses.
    DEBUG_ERRORS: bool = True


settings = Settings()
