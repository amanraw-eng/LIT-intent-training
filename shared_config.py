"""Shared configuration helpers for the pipeline and training scripts.

Secrets belong in the repository-root ``.env`` (which is gitignored).
Non-secret defaults belong in ``shared_settings.py`` and are safe to commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(path: str) -> str:
    """Resolve a relative project path, such as a local credential path."""
    candidate = Path(path).expanduser()
    return str(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate)


def load_environment(*, legacy_env: Path | None = None) -> None:
    """Load the common .env without replacing shell variables."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if legacy_env:
        load_dotenv(legacy_env, override=False)


from shared_settings import SETTINGS


def section(name: str) -> dict[str, Any]:
    """Return one config section with a clear error for a typo."""
    try:
        return SETTINGS[name]
    except KeyError as error:
        raise KeyError(f"Missing settings section {name!r} in shared_settings.py") from error
