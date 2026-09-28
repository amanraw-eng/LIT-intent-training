"""Deployment-specific settings, loaded from the environment / .env.

Only real secrets and values that legitimately differ per deployment live
here (which HF model to serve, credentials, where to listen, log
verbosity). Fixed, code-level defaults live in constants.py instead.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to this file (repo_root/src/app/config/settings.py -> repo_root/.env)
# instead of the process CWD, so `.env` (including PORT/HOST) loads the same
# way whether run via `uv run python -m app.main` from anywhere, a plain
# `python -m app.main`, or systemd.
_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    HF_MODEL_REPO: str = "amn-raw/whisper-small-intent17-classifier"
    HF_TOKEN: str = ""

    LOG_LEVEL: str = "INFO"
    # Set False in production if you don't want exception details exposed
    # in API error responses.
    DEBUG_ERRORS: bool = True

    # Concurrent requests are batched into a single GPU forward pass instead
    # of one call per request. BATCH_MAX_WAIT_MS bounds how long the first
    # request in a batch waits for others to join it before the batch is run
    # as-is; BATCH_MAX_SIZE caps how many join regardless of wait time. Tune
    # these against your GPU's memory/throughput once you have real traffic.
    BATCH_MAX_SIZE: int = 32
    BATCH_MAX_WAIT_MS: float = 10.0

    # Audio decode/preprocessing (CPU-bound: spawns ffmpeg, computes a mel
    # spectrogram) runs in a thread pool of this size, independent of and
    # concurrent with GPU batching above. None uses Starlette/anyio's default.
    DECODE_THREAD_POOL_SIZE: int | None = None


settings = Settings()
