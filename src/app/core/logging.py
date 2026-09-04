"""Logging setup for the API service - one place to configure format/level."""

from __future__ import annotations

import logging

from .config import settings


def configure_logging() -> logging.Logger:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("whisper-intent-api")


logger = configure_logging()
