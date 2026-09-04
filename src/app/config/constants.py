"""Fixed, code-level defaults - not meant to vary per deployment.

If a value here ever needs to differ between deployments, move it to
settings.py (env-driven) instead of reaching for an environment variable
override here.
"""

from __future__ import annotations

MODEL_TYPE = "small"
DEVICE = "cuda"

TARGET_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION_SECONDS = 0.5
MAX_AUDIO_DURATION_SECONDS = 10.0

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_TOP_K = 20
