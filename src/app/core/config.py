"""Runtime configuration for the intent-classification API.

Self-contained: this service does not import from any other package in the
repository (pipeline/, data_prep/, shared_config.py, ...) so it can be built
and deployed on its own. Only real secrets / deployment-specific values are
read from the environment (see .env.example) - everything else is a plain
constant here, carried over as-is from the original
intent-training/API_CONFIG.py so runtime behavior is unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parents[2]
load_dotenv(SRC_DIR / ".env")


class Settings:
    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    MODEL_TYPE: str = "small"
    DEVICE: str = "cuda"
    TARGET_SAMPLE_RATE: int = 16000
    MIN_AUDIO_DURATION_SECONDS: float = 0.5
    MAX_AUDIO_DURATION_SECONDS: float = 10.0

    # Hugging Face model source - env-overridable (deployment picks which
    # trained model to serve), same as the original API_CONFIG.py.
    HF_MODEL_REPO: str = os.environ.get("HF_MODEL_REPO", "amn-raw/whisper-small-intent17-classifier")
    HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024
    MAX_TOP_K: int = 20

    # ------------------------------------------------------------------
    # Ops
    # ------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    # Set False in production if you don't want exception details exposed
    # in API error responses.
    DEBUG_ERRORS: bool = True


settings = Settings()
