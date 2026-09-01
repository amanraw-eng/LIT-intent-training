from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from datasets import Audio, DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download

from . import config


# ============================================================
# CONFIG
# ============================================================

SOURCE_REPO = "kapturecx/call-transcript-intent-data"
DEST_REPO = "kapturecx/call-transcript-intent-data-v2"

SPLITS = (
    "train",
    "validation",
    "eval",
)

HF_TOKEN = (
    getattr(config, "HF_TOKEN", None)
    or os.getenv("HF_TOKEN")
)

CHECKPOINT_PATH = getattr(
    config,
    "RELABEL_CHECKPOINT",
    "relabel_checkpoint.jsonl",
)

HF_AUDIO_PREFIX = getattr(
    config,
    "RELABEL_HF_AUDIO_PREFIX",
    "audio",
)

AUDIO_DOWNLOAD_BATCH_SIZE = int(
    getattr(
        config,
        "RELABEL_PUSH_AUDIO_BATCH_SIZE",
        500,
    )
)

HF_PRIVATE = bool(
    getattr(
        config,
        "RELABEL_HF_PRIVATE",
        False,
    )
)


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(path: str) -> dict[tuple[str, int], dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

                key = (
                    record["split"],
                    int(record["dataset_index"]),
                )

                checkpoint[key] = record

            except Exception as exc:
                raise RuntimeError(
                    f"Invalid checkpoint line "
                    f"{line_number}: {exc}"
                ) from exc

    print(
        f"[push] checkpoint rows: {len(checkpoint):,}"
    )

    return checkpoint


# ============================================================
# LOAD ORIGINAL HF DATASET
# ============================================================

def load_source_dataset() -> DatasetDict:
    print(
        f"[push] loading source dataset: {SOURCE_REPO}"
    )

    dataset = load_dataset(
        SOURCE_REPO,
        token=HF_TOKEN,
    )

    if not isinstance(dataset, DatasetDict):
        raise RuntimeError(
            f"Expected DatasetDict, got {type(dataset)}"
        )

    for split in SPLITS:
        if split not in dataset:
            raise RuntimeError(
                f"Missing split '{split}'. "
                f"Available: {list(dataset.keys())}"
            )

        # Never automatically decode stale local paths.
        dataset[split] = dataset[split].cast_column(
            "audio",
            Audio(decode=False),
        )

        print(
            f"[push] {split}: "
            f"{len(dataset[split]):,} rows"
        )

    return dataset


# ============================================================
# APPLY RELABEL CHECKPOINT
# ============================================================

def apply_checkpoint(
    dataset: DatasetDict,
    checkpoint: dict[tuple[str, int], dict],
) -> DatasetDict:

    updated = {}

    for split in SPLITS:
        ds = dataset[split]

        intents = list(ds["intent"])

        if "transcript_reviewed" in ds.column_names:
            reviewed = list(
                ds["transcript_reviewed"]
            )
        else:
            reviewed = list(
                ds["transcript"]
            )

        changed = 0

        for index in range(len(ds)):
            record = checkpoint.get(
                (split, index)
            )

            if record is None:
                continue

            intents[index] = record["intent"]
            reviewed[index] = record[
                "transcript_reviewed"
            ]

            changed += 1

        print(
            f"[push] {split}: "
            f"applied checkpoint to "
            f"{changed:,} rows"
        )

        if "transcript_reviewed" in ds.column_names:
            ds = ds.remove_columns(
                ["transcript_reviewed"]
            )

        ds = ds.add_column(
            "transcript_reviewed",
            reviewed,
        )

        ds = ds.remove_columns(
            ["intent"]
        )

        ds = ds.add_column(
            "intent",
            intents,
        )

        updated[split] = ds

    return DatasetDict(updated)


# ============================================================
# HUB AUDIO INDEX
# ============================================================

def build_hub_audio_index():
    """
    Build:
        basename -> [repo-relative filenames]

    We deliberately search the whole repository instead of assuming
    the audio is under a particular directory.
    """

    print(
        "[push] listing audio files in source HF repo..."
    )

    api = HfApi(
        token=HF_TOKEN
    )

    files = api.list_repo_files(
        repo_id=SOURCE_REPO,
        repo_type="dataset",
    )

    audio_files = []

    for filename in files:
        if filename.lower().endswith(
            (
                ".wav",
                ".mp3",
                ".m4a",
                ".flac",
                ".ogg",
                ".opus",
                ".webm",
                ".mp4",
            )
        ):
            audio_files.append(filename)

    by_basename = defaultdict(list)

    for filename in audio_files:
        by_basename[
            Path(filename).name
        ].append(filename)

    print(
        f"[push] Hub audio files indexed: "
        f"{len(audio_files):,}"
    )

    return by_basename


# ============================================================
# AUDIO RESOLUTION
# ============================================================

def get_local_basename(
    audio: dict | None,
) -> str | None:

    if not isinstance(audio, dict):
        return None

    path = audio.get("path")

    if not path:
        return None

    return Path(
        str(path)
    ).name


def resolve_hub_filename(
    audio: dict | None,
    basename_index,
) -> str | None:

    basename = get_local_basename(
        audio
    )

    if not basename:
        return None

    matches = basename_index.get(
        basename,
        [],
    )

    if not matches:
        return None

    if len(matches) == 1:
        return matches[0]

    preferred = (
        f"{HF_AUDIO_PREFIX}/{basename}"
    )

    if preferred in matches:
        return preferred

    return matches[0]


def resolve_audio_from_hub(
    audio: dict | None,
    basename_index,
) -> tuple[dict | None, str | None]:

    # --------------------------------------------------------
    # Case 1: HF dataset already has actual embedded bytes.
    # --------------------------------------------------------

    if isinstance(audio, dict):
        embedded = audio.get("bytes")

        if embedded:
            return (
                {
                    "bytes": bytes(embedded),
                    "path": None,
                },
                None,
            )

    # --------------------------------------------------------
    # Case 2: stale local path, but matching file exists
    # somewhere in the HF repository.
    # --------------------------------------------------------

    hub_filename = resolve_hub_filename(
        audio,
        basename_index,
    )

    if not hub_filename:
        return None, None

    try:
        cached_path = hf_hub_download(
            repo_id=SOURCE_REPO,
            filename=hub_filename,
            repo_type="dataset",
            token=HF_TOKEN,
        )

        return (
            {
                "path": cached_path,
                "bytes": None,
            },
            hub_filename,
        )

    except Exception as exc:
        print(
            f"[push] failed to download Hub audio "
            f"{hub_filename}: {exc}"
        )

        return None, hub_filename


# ============================================================
# REBUILD AUDIO COLUMN
# ============================================================

def rebuild_audio(
    dataset: DatasetDict,
    basename_index,
):
    updated = {}
    missing = []

    for split in SPLITS:
        ds = dataset[split]

        total = len(ds)

        print(
            f"[push] resolving audio for "
            f"{split}: {total:,} rows"
        )

        audio_values = []

        for start in range(
            0,
            total,
            AUDIO_DOWNLOAD_BATCH_SIZE,
        ):
            end = min(
                start + AUDIO_DOWNLOAD_BATCH_SIZE,
                total,
            )

            print(
                f"[push] {split} audio "
                f"{end:,}/{total:,}"
            )

            for index in range(
                start,
                end,
            ):
                row = ds[index]
                audio = row.get("audio")

                resolved_audio, hub_filename = (
                    resolve_audio_from_hub(
                        audio,
                        basename_index,
                    )
                )

                if resolved_audio is None:
                    missing.append(
                        {
                            "split": split,
                            "dataset_index": index,
                            "id": str(
                                row.get(
                                    "id",
                                    index,
                                )
                            ),
                            "local_audio_path": (
                                audio.get("path")
                                if isinstance(
                                    audio,
                                    dict,
                                )
                                else None
                            ),
                            "hub_filename": hub_filename,
                        }
                    )

                    audio_values.append(None)
                else:
                    audio_values.append(
                        resolved_audio
                    )

        # Remove stale audio feature.
        ds = ds.remove_columns(
            ["audio"]
        )
        
        ds = ds.add_column(
            "audio",
            audio_values,
        )
        
        ds = ds.cast_column(
            "audio",
            Audio(),
        )

        # Keep audio first.
        columns = list(
            ds.column_names
        )

        columns.remove("audio")
        columns.insert(0, "audio")

        ds = ds.select_columns(
            columns
        )

        updated[split] = ds

    return (
        DatasetDict(updated),
        missing,
    )


# ============================================================
# REMOVE ROWS WITHOUT AUDIO
# ============================================================

def remove_missing_audio_rows(
    dataset: DatasetDict,
    missing: list[dict],
) -> DatasetDict:

    missing_keys = {
        (
            item["split"],
            int(item["dataset_index"]),
        )
        for item in missing
    }

    updated = {}

    total_removed = 0

    for split in SPLITS:
        ds = dataset[split]

        keep_indices = [
            index
            for index in range(len(ds))
            if (
                split,
                index,
            ) not in missing_keys
        ]

        removed = (
            len(ds)
            - len(keep_indices)
        )

        if removed:
            print(
                f"[push] {split}: "
                f"removing {removed:,} "
                f"rows with unavailable audio"
            )

        total_removed += removed

        updated[split] = ds.select(
            keep_indices
        )

    print(
        f"[push] total rows removed for "
        f"unavailable audio: "
        f"{total_removed:,}"
    )

    return DatasetDict(updated)


# ============================================================
# LOG MISSING AUDIO
# ============================================================

def save_missing_audio(
    missing: list[dict],
    path: str = "push_missing_audio.jsonl",
):

    if not missing:
        return

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        for record in missing:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"[push] missing audio log: "
        f"{path}"
    )


# ============================================================
# VALIDATE FINAL DATASET
# ============================================================

def validate_final_dataset(
    dataset: DatasetDict,
):
    """
    Validate the final dataset without assuming optional columns
    such as chunk_idx exist in every source split.

    Required:
        audio
        intent
        transcript
        transcript_reviewed
        id

    All other source columns are preserved as-is.
    """

    required_columns = {
        "audio",
        "intent",
        "transcript",
        "transcript_reviewed",
        "id",
    }

    for split in SPLITS:
        ds = dataset[split]

        actual_columns = set(
            ds.column_names
        )

        missing_columns = (
            required_columns
            - actual_columns
        )

        if missing_columns:
            raise RuntimeError(
                f"{split} missing required columns: "
                f"{sorted(missing_columns)}\n"
                f"Actual columns: {ds.column_names}"
            )

        if "original_intent" in actual_columns:
            raise RuntimeError(
                f"{split} unexpectedly contains "
                f"original_intent column."
            )

        null_audio = 0

        for index in range(
            len(ds)
        ):
            audio = ds[index]["audio"]

            if audio is None:
                null_audio += 1

                if null_audio <= 10:
                    print(
                        f"[push] WARNING null audio | "
                        f"{split}:{index} | "
                        f"id={ds[index]['id']}"
                    )

        if null_audio:
            raise RuntimeError(
                f"{split} still has "
                f"{null_audio:,} rows with missing audio."
            )

        print(
            f"[push] {split}: validation OK "
            f"({len(ds):,} rows)"
        )

        print(
            f"[push] {split} columns: "
            f"{ds.column_names}"
        )


# ============================================================
# PUSH
# ============================================================

def push_dataset(
    dataset: DatasetDict,
    repo_id: str,
):

    print(
        f"[push] pushing to: {repo_id}"
    )

    dataset.push_to_hub(
        repo_id=repo_id,
        token=HF_TOKEN,
        private=HF_PRIVATE,
    )

    print(
        "[push] PUSH COMPLETE"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Apply relabel checkpoint and push a new HF "
            "dataset using audio available from the source "
            "HF repository."
        )
    )

    parser.add_argument(
        "--repo",
        default=DEST_REPO,
        help=(
            "Destination HF repository."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_PATH,
    )

    parser.add_argument(
        "--no-push",
        action="store_true",
        help=(
            "Build and validate only."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional local save_to_disk path."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load checkpoint.
    # --------------------------------------------------------

    checkpoint = load_checkpoint(
        args.checkpoint
    )

    # --------------------------------------------------------
    # Load source HF dataset.
    # --------------------------------------------------------

    dataset = load_source_dataset()

    # --------------------------------------------------------
    # Apply new intents/reviewed transcripts.
    #
    # This happens BEFORE dropping any missing-audio rows,
    # because checkpoint dataset_index values refer to the
    # original dataset.
    # --------------------------------------------------------

    dataset = apply_checkpoint(
        dataset,
        checkpoint,
    )

    # --------------------------------------------------------
    # Resolve actual Hub audio.
    # --------------------------------------------------------

    basename_index = (
        build_hub_audio_index()
    )

    (
        dataset,
        missing,
    ) = rebuild_audio(
        dataset,
        basename_index,
    )

    # --------------------------------------------------------
    # Missing audio is allowed.
    # Log and remove those rows.
    # --------------------------------------------------------

    if missing:

        print(
            f"[push] unavailable audio rows: "
            f"{len(missing):,}"
        )

        save_missing_audio(
            missing
        )

        dataset = remove_missing_audio_rows(
            dataset,
            missing,
        )

    else:

        print(
            "[push] all audio rows resolved."
        )

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

    validate_final_dataset(
        dataset
    )

    # --------------------------------------------------------
    # Optional local copy.
    # --------------------------------------------------------

    if args.output:

        print(
            f"[push] saving local dataset: "
            f"{args.output}"
        )

        dataset.save_to_disk(
            args.output
        )

        print(
            "[push] local dataset saved."
        )

    # --------------------------------------------------------
    # Push.
    # --------------------------------------------------------

    if args.no_push:

        print(
            "[push] --no-push specified. "
            "Nothing uploaded."
        )

        return

    push_dataset(
        dataset,
        args.repo,
    )


if __name__ == "__main__":
    main()