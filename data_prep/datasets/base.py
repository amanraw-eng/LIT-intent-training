"""Base abstraction for any source of intent-labeled (id, audio, transcript,
intent) rows - a local JSONL file or a Hugging Face dataset.

Subclasses only implement iteration and length; column-name configuration
and id fallback are handled here so callers always see the same canonical
shape (record["id"], record["audio"], record["transcript"], record["intent"])
regardless of the underlying source or its actual column names.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class IntentColumns:
    """Maps canonical field names to this source's actual column names."""

    id: str = "id"
    audio: str = "audio"
    transcript: str = "transcript"
    intent: str = "intent"


class IntentRecord(dict):
    """One normalized row: canonical id/audio/transcript/intent keys, plus the
    original row unchanged under "raw" (so task-specific extra columns, e.g.
    conversation_id or recording_url, are never lost)."""


class IntentDataSource(ABC):
    """Base class for a readable collection of intent-labeled rows.

    Default column names are "id", "audio", "transcript", "intent" - pass
    `id_column=`, `audio_column=`, `transcript_column=`, `intent_column=` to
    a subclass constructor to point at differently-named columns.
    """

    def __init__(
        self,
        *,
        id_column: str = "id",
        audio_column: str = "audio",
        transcript_column: str = "transcript",
        intent_column: str = "intent",
    ):
        self.columns = IntentColumns(
            id=id_column,
            audio=audio_column,
            transcript=transcript_column,
            intent=intent_column,
        )

    def _normalize(self, raw_row: dict, *, fallback_id: Any) -> IntentRecord:
        return IntentRecord(
            id=raw_row.get(self.columns.id, fallback_id),
            audio=raw_row.get(self.columns.audio),
            transcript=raw_row.get(self.columns.transcript),
            intent=raw_row.get(self.columns.intent),
            raw=raw_row,
        )

    @abstractmethod
    def __iter__(self) -> Iterator[IntentRecord]: ...

    @abstractmethod
    def __len__(self) -> int: ...
