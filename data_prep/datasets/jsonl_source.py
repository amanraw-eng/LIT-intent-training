from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import IntentDataSource, IntentRecord


class JsonlIntentDataSource(IntentDataSource):
    """Reads intent-labeled rows from a local JSONL file (one JSON object per
    line), e.g. pipeline/data.jsonl-style generation output."""

    def __init__(self, path: str | Path, **column_kwargs):
        super().__init__(**column_kwargs)
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL dataset not found: {self.path}")

    def __iter__(self) -> Iterator[IntentRecord]:
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                yield self._normalize(json.loads(line), fallback_id=i)

    def __len__(self) -> int:
        with open(self.path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
