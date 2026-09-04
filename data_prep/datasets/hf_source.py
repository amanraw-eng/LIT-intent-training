from __future__ import annotations

from typing import Iterator

from .base import IntentDataSource, IntentRecord
from .. import config


class HFIntentDataSource(IntentDataSource):
    """Reads intent-labeled rows from a Hugging Face dataset - either a hub
    repo id (downloaded via `datasets.load_dataset`) or a local
    `datasets.save_to_disk` directory (via `load_from_disk`)."""

    def __init__(
        self,
        repo_id_or_path: str,
        *,
        split: str = "train",
        from_disk: bool = False,
        token: str | None = None,
        decode_audio: bool = True,
        **column_kwargs,
    ):
        super().__init__(**column_kwargs)
        from datasets import Audio, load_dataset, load_from_disk

        dataset = load_from_disk(repo_id_or_path) if from_disk else load_dataset(
            repo_id_or_path, token=token or config.HF_TOKEN
        )

        if hasattr(dataset, "keys") and split in dataset:
            dataset = dataset[split]

        if not decode_audio and self.columns.audio in dataset.column_names:
            # Skip eager audio decoding - callers that only need transcripts/
            # metadata (or that resolve audio bytes themselves) shouldn't pay
            # for it.
            dataset = dataset.cast_column(self.columns.audio, Audio(decode=False))

        self.dataset = dataset

    def __iter__(self) -> Iterator[IntentRecord]:
        for i, row in enumerate(self.dataset):
            yield self._normalize(row, fallback_id=i)

    def __len__(self) -> int:
        return len(self.dataset)
