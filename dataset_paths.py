"""Portable JSONL audio-path helpers.

JSONL stores ``chunk_path`` relative to an audio folder. Callers supply that
folder when reading the manifest, making the same JSONL usable on any machine.
"""
from __future__ import annotations

from pathlib import Path


def resolve_chunk_path(chunk_path: str, audio_dir: str | Path | None) -> Path:
    """Resolve a portable relative chunk_path; accepts legacy absolute paths."""
    if not chunk_path:
        raise ValueError("chunk_path cannot be empty")
    path = Path(chunk_path)
    if path.is_absolute():
        return path
    if audio_dir is None:
        raise ValueError("audio_dir is required when JSONL chunk_path is relative")
    root = Path(audio_dir).expanduser().resolve()
    resolved = (root / path).resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"chunk_path escapes audio_dir: {chunk_path!r}")
    return resolved


def make_relative_chunk_path(chunk_path: str, audio_dir: str | Path) -> str:
    """Convert an absolute or relative path to a normalized portable path."""
    root = Path(audio_dir).expanduser().resolve()
    resolved = resolve_chunk_path(chunk_path, root)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"audio file is outside audio_dir: {resolved}") from error
