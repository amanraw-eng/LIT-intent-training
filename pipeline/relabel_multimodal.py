from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import time
from pathlib import Path
from typing import Literal

from datasets import Audio, DatasetDict, load_dataset
from google import genai
from google.genai import types
from google.oauth2 import service_account
from pydantic import BaseModel, Field

from . import config
from .relabel_prompt import SYSTEM_PROMPT


# ============================================================
# CONFIG
# ============================================================

HF_DATASET_REPO = "kapturecx/call-transcript-intent-data"

HF_DATASET_SPLITS = (
    "train",
    "validation",
    "eval",
)

BATCH_SIZE = 10

MAX_CONCURRENCY = int(
    getattr(
        config,
        "RELABEL_MAX_CONCURRENCY",
        4,
    )
)

MAX_RETRIES = int(
    getattr(
        config,
        "RELABEL_MAX_RETRIES",
        4,
    )
)

RETRY_DELAY_S = float(
    getattr(
        config,
        "RELABEL_RETRY_DELAY_S",
        2.0,
    )
)

GEMINI_KEY_PATH = config.GEMINI_KEY_PATH
GEMINI_LOCATION = config.GEMINI_LOCATION
GEMINI_MODEL = config.GEMINI_MODEL

HF_TOKEN = (
    getattr(config, "HF_TOKEN", None)
    or os.getenv("HF_TOKEN")
)

# ------------------------------------------------------------
# Transcript handling
# ------------------------------------------------------------

LONG_TRANSCRIPT_LIMIT = int(
    getattr(
        config,
        "RELABEL_LONG_TRANSCRIPT_LIMIT",
        100,
    )
)

GENERATED_REVIEW_MAX_CHARS = int(
    getattr(
        config,
        "RELABEL_GENERATED_REVIEW_MAX_CHARS",
        500,
    )
)

# ------------------------------------------------------------
# Pathological ASR filtering
# ------------------------------------------------------------

PATHOLOGICAL_MAX_CHARS = int(
    getattr(
        config,
        "RELABEL_PATHOLOGICAL_MAX_CHARS",
        300,
    )
)

PATHOLOGICAL_UNIQUE_RATIO = float(
    getattr(
        config,
        "RELABEL_PATHOLOGICAL_UNIQUE_RATIO",
        0.20,
    )
)

PATHOLOGICAL_MAX_RUN = int(
    getattr(
        config,
        "RELABEL_PATHOLOGICAL_MAX_RUN",
        8,
    )
)

# ------------------------------------------------------------
# Gemini output limits
# ------------------------------------------------------------

REVIEW_MAX_OUTPUT_TOKENS = int(
    getattr(
        config,
        "RELABEL_REVIEW_MAX_OUTPUT_TOKENS",
        4096,
    )
)

INTENT_ONLY_MAX_OUTPUT_TOKENS = int(
    getattr(
        config,
        "RELABEL_INTENT_ONLY_MAX_OUTPUT_TOKENS",
        1024,
    )
)

# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

CHECKPOINT_PATH = getattr(
    config,
    "RELABEL_CHECKPOINT",
    "relabel_checkpoint.jsonl",
)

MISSING_AUDIO_PATH = getattr(
    config,
    "RELABEL_MISSING_AUDIO",
    "relabel_missing_audio.jsonl",
)

TEST_OUTPUT = getattr(
    config,
    "RELABEL_TEST_OUTPUT",
    "test_relabel.json",
)

FULL_OUTPUT = getattr(
    config,
    "RELABEL_FULL_OUTPUT",
    "call-transcript-intent-data-relabeled",
)


# ============================================================
# CLOSED TAXONOMY
# ============================================================

IntentName = Literal[
    "AFFIRMATIVE_ACKNOWLEDGEMENT",
    "NEGATIVE_ACKNOWLEDGEMENT",
    "IDENTITY_CONFIRMED",
    "THIRD_PARTY_AVAILABLE",
    "THIRD_PARTY_UNAVAILABLE",
    "PAY_NOW_AGREE",
    "PAY_LATER_AGREE",
    "PAID_ALREADY",
    "REFUSE_TO_PAY",
    "NO_PAYMENT_REASON",
    "END_CALL",
    "DO_NOT_CALL",
    "CALL_DEFER",
    "BACKCHANNEL_OR_NOISE",
    "UNCLEAR_INPUT",
]


class RelabelItem(BaseModel):
    """
    No max_length here.

    Original long Whisper transcripts must be allowed to pass through
    unchanged when a row is intentionally not reviewed.
    """

    index: int
    transcript_reviewed: str
    intent: IntentName
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class RelabelBatchResult(BaseModel):
    items: list[RelabelItem]


class IntentOnlyItem(BaseModel):
    index: int
    intent: IntentName
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class IntentOnlyBatchResult(BaseModel):
    items: list[IntentOnlyItem]


# ============================================================
# GEMINI CLIENT
# ============================================================

def build_gemini_client():
    if not GEMINI_KEY_PATH:
        raise RuntimeError(
            "config.GEMINI_KEY_PATH is not configured."
        )

    key_path = Path(
        GEMINI_KEY_PATH
    ).expanduser()

    if not key_path.exists():
        raise FileNotFoundError(
            f"Gemini service account file not found: "
            f"{key_path}"
        )

    with open(
        key_path,
        encoding="utf-8",
    ) as f:
        service_info = json.load(f)

    project_id = service_info.get(
        "project_id"
    )

    if not project_id:
        raise RuntimeError(
            "Gemini service account JSON does not contain project_id."
        )

    credentials = (
        service_account.Credentials.from_service_account_file(
            str(key_path),
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform"
            ],
        )
    )

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=GEMINI_LOCATION,
        credentials=credentials,
    )

    print(
        f"[relabel] Gemini model    : {GEMINI_MODEL}"
    )
    print(
        f"[relabel] Gemini location : {GEMINI_LOCATION}"
    )

    return client


# ============================================================
# DATASET
# ============================================================

def load_hf_dataset() -> DatasetDict:
    print(
        f"[relabel] loading HF dataset: "
        f"{HF_DATASET_REPO}"
    )

    dataset = load_dataset(
        HF_DATASET_REPO,
        token=HF_TOKEN,
    )

    if not isinstance(
        dataset,
        DatasetDict,
    ):
        raise RuntimeError(
            f"Expected DatasetDict, got {type(dataset)}"
        )

    missing = [
        split
        for split in HF_DATASET_SPLITS
        if split not in dataset
    ]

    if missing:
        raise RuntimeError(
            f"Missing expected splits: {missing}. "
            f"Available: {list(dataset.keys())}"
        )

    for split in HF_DATASET_SPLITS:

        if "audio" not in dataset[split].column_names:
            raise RuntimeError(
                f"'audio' column missing from split "
                f"'{split}'. Columns: "
                f"{dataset[split].column_names}"
            )

        # Do not let torchcodec automatically decode every row.
        dataset[split] = dataset[split].cast_column(
            "audio",
            Audio(decode=False),
        )

        print(
            f"[relabel] split={split}, "
            f"rows={len(dataset[split])}"
        )

    return dataset


# ============================================================
# CHECKPOINT
# ============================================================

def checkpoint_key(
    split: str,
    dataset_index: int,
) -> str:
    return f"{split}:{dataset_index}"


def load_checkpoint(
    path: str,
) -> dict[str, dict]:

    completed = {}

    if not os.path.exists(path):
        print(
            f"[relabel] checkpoint not found: "
            f"{path}"
        )
        return completed

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

                key = checkpoint_key(
                    record["split"],
                    int(
                        record["dataset_index"]
                    ),
                )

                completed[key] = record

            except Exception as exc:

                raise RuntimeError(
                    f"Invalid checkpoint line "
                    f"{line_number}: {exc}"
                ) from exc

    print(
        f"[relabel] checkpoint rows: "
        f"{len(completed):,}"
    )

    return completed


def append_checkpoint(
    path: str,
    records: list[dict],
) -> None:

    if not records:
        return

    parent = os.path.dirname(
        os.path.abspath(path)
    )

    if parent:
        os.makedirs(
            parent,
            exist_ok=True,
        )

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        f.flush()
        os.fsync(
            f.fileno()
        )


# ============================================================
# MISSING AUDIO LOG
# ============================================================

def append_missing_audio(
    path: str,
    records: list[dict],
) -> None:

    if not records:
        return

    parent = os.path.dirname(
        os.path.abspath(path)
    )

    if parent:
        os.makedirs(
            parent,
            exist_ok=True,
        )

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        f.flush()
        os.fsync(
            f.fileno()
        )


# ============================================================
# AUDIO
# ============================================================

def get_audio_reference(
    audio: dict,
) -> str | None:

    if not isinstance(
        audio,
        dict,
    ):
        return None

    path = audio.get(
        "path"
    )

    return str(path) if path else None


def audio_file_exists(
    audio: dict,
) -> bool:

    if not isinstance(
        audio,
        dict,
    ):
        return False

    if audio.get("bytes"):
        return True

    path = audio.get(
        "path"
    )

    if not path:
        return False

    return Path(
        os.path.expanduser(
            str(path)
        )
    ).exists()


def mime_type_from_path(
    path: str | None,
) -> str:

    if not path:
        return "audio/wav"

    suffix = (
        Path(path)
        .suffix
        .lower()
    )

    mapping = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".mpeg": "audio/mpeg",
        ".mpga": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".webm": "audio/webm",
    }

    return mapping.get(
        suffix,
        "audio/wav",
    )


def load_audio_bytes(
    audio: dict,
) -> tuple[bytes, str]:

    if not isinstance(
        audio,
        dict,
    ):
        raise FileNotFoundError(
            "Invalid audio object."
        )

    embedded = audio.get(
        "bytes"
    )

    if embedded:
        return (
            bytes(embedded),
            mime_type_from_path(
                audio.get("path")
            ),
        )

    path = audio.get(
        "path"
    )

    if not path:
        raise FileNotFoundError(
            "Audio has neither bytes nor path."
        )

    path = Path(
        os.path.expanduser(
            str(path)
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {path}"
        )

    with open(
        path,
        "rb",
    ) as f:

        data = f.read()

    return (
        data,
        mime_type_from_path(
            str(path)
        ),
    )


# ============================================================
# TRANSCRIPT FILTERS
# ============================================================

def is_long_transcript(
    transcript: str,
) -> bool:

    return len(
        (transcript or "").strip()
    ) > LONG_TRANSCRIPT_LIMIT


def is_pathological_transcript(
    text: str,
) -> bool:
    """
    Conservative filter for obvious ASR loops/garbage.

    Examples:
        तो तो तो तो तो तो...
        तव तव तव तव...
    """

    text = (
        text or ""
    ).strip()

    if not text:
        return True

    if len(text) > PATHOLOGICAL_MAX_CHARS:
        return True

    words = text.split()

    if len(words) < 12:
        return False

    unique_ratio = (
        len(set(words))
        / max(
            len(words),
            1,
        )
    )

    if unique_ratio <= PATHOLOGICAL_UNIQUE_RATIO:
        return True

    longest_run = 1
    current_run = 1

    for i in range(
        1,
        len(words),
    ):

        if words[i] == words[i - 1]:

            current_run += 1
            longest_run = max(
                longest_run,
                current_run,
            )

        else:

            current_run = 1

    return longest_run >= PATHOLOGICAL_MAX_RUN


def is_suspicious_generated_review(
    text: str,
) -> bool:

    text = (
        text or ""
    ).strip()

    if not text:
        return False

    if len(text) > GENERATED_REVIEW_MAX_CHARS:
        return True

    words = text.split()

    if len(words) < 12:
        return False

    unique_ratio = (
        len(set(words))
        / max(
            len(words),
            1,
        )
    )

    if unique_ratio <= 0.15:
        return True

    longest_run = 1
    current_run = 1

    for i in range(
        1,
        len(words),
    ):

        if words[i] == words[i - 1]:

            current_run += 1
            longest_run = max(
                longest_run,
                current_run,
            )

        else:

            current_run = 1

    return longest_run >= 8


# ============================================================
# ROW SELECTION
# ============================================================

def select_rows(
    dataset_dict: DatasetDict,
    checkpoint: dict[str, dict],
    limit: int | None = None,
):
    """
    Returns:
        normal_rows
        missing_audio
        pathological_rows

    Already checkpointed rows are always skipped.
    """

    normal_rows = []
    missing_audio = []
    pathological_rows = []

    selected_count = 0

    for split in HF_DATASET_SPLITS:

        dataset = dataset_dict[
            split
        ]

        for dataset_index in range(
            len(dataset)
        ):

            key = checkpoint_key(
                split,
                dataset_index,
            )

            # NEVER RERUN CHECKPOINTED ROW.
            if key in checkpoint:
                continue

            row = dataset[
                dataset_index
            ]

            transcript = (
                row.get("transcript")
                or ""
            ).strip()

            if not transcript:
                continue

            audio = row.get(
                "audio"
            )

            metadata = {
                "split": split,
                "dataset_index": dataset_index,
                "id": str(
                    row.get(
                        "id",
                        dataset_index,
                    )
                ),
                "oid": (
                    str(row["oid"])
                    if row.get("oid") is not None
                    else None
                ),
                "conversation_id": (
                    str(
                        row[
                            "conversation_id"
                        ]
                    )
                    if row.get(
                        "conversation_id"
                    ) is not None
                    else None
                ),
                "recording_url": row.get(
                    "recording_url"
                ),
                "audio_reference": (
                    get_audio_reference(
                        audio
                    )
                ),
                "transcript": transcript,
                "old_intent": row[
                    "intent"
                ],
            }

            if not audio_file_exists(
                audio
            ):

                missing_audio.append(
                    {
                        **metadata,
                        "reason": (
                            "audio bytes unavailable "
                            "and local audio file does not exist"
                        ),
                    }
                )

                continue

            row_data = {
                **metadata,
                "audio": audio,
            }

            if is_pathological_transcript(
                transcript
            ):
                pathological_rows.append(
                    row_data
                )
            else:
                normal_rows.append(
                    row_data
                )

            selected_count += 1

            if (
                limit is not None
                and selected_count >= limit
            ):
                return (
                    normal_rows,
                    missing_audio,
                    pathological_rows,
                )

    return (
        normal_rows,
        missing_audio,
        pathological_rows,
    )


# ============================================================
# PROMPTS
# ============================================================

def build_review_batch_text(
    batch: list[dict],
) -> str:

    lines = [
        "Classify each customer audio independently.",
        "Use the audio as the source of truth.",
        "",
        "The supplied transcript is usually correct.",
        "Correct only genuine ASR mistakes.",
        "Do NOT expand the transcript.",
        "Do NOT paraphrase.",
        "Do NOT invent words.",
        "Keep transcript_reviewed close to what was actually spoken.",
        "",
        "Return exactly one result per item.",
        "",
    ]

    for local_index, item in enumerate(
        batch
    ):

        lines.extend(
            [
                f"===== ITEM {local_index} =====",
                f"id: {item['id']}",
                f"current_transcript: {item['transcript']}",
                "",
            ]
        )

    return "\n".join(
        lines
    )


def build_intent_only_batch_text(
    batch: list[dict],
) -> str:

    lines = [
        "Classify each customer audio independently.",
        "Do NOT rewrite or correct transcripts.",
        "Only determine the intent.",
        "Use the audio as the source of truth.",
        "When uncertain, use UNCLEAR_INPUT.",
        "",
    ]

    for local_index, item in enumerate(
        batch
    ):

        lines.extend(
            [
                f"===== ITEM {local_index} =====",
                f"id: {item['id']}",
                f"current_transcript: {item['transcript']}",
                "",
            ]
        )

    return "\n".join(
        lines
    )


INTENT_ONLY_SYSTEM_PROMPT = r"""
You are a HIGH-PRECISION intent classifier for loan/EMI/payment
customer phone calls.

Classify ONLY the current customer utterance.

Use the audio as the source of truth.

No missing context inference.

False positives are worse than UNCLEAR_INPUT.

Allowed intents:

AFFIRMATIVE_ACKNOWLEDGEMENT
NEGATIVE_ACKNOWLEDGEMENT
IDENTITY_CONFIRMED
THIRD_PARTY_AVAILABLE
THIRD_PARTY_UNAVAILABLE
PAY_NOW_AGREE
PAY_LATER_AGREE
PAID_ALREADY
REFUSE_TO_PAY
NO_PAYMENT_REASON
END_CALL
DO_NOT_CALL
CALL_DEFER
BACKCHANNEL_OR_NOISE
UNCLEAR_INPUT

Never invent an intent.

Return ONLY valid JSON.
"""


# ============================================================
# GEMINI REVIEW
# ============================================================

def classify_review_batch(
    client,
    batch: list[dict],
) -> RelabelBatchResult:

    parts = [
        types.Part.from_text(
            text=build_review_batch_text(
                batch
            )
        )
    ]

    for local_index, item in enumerate(
        batch
    ):

        audio_bytes, mime_type = (
            load_audio_bytes(
                item["audio"]
            )
        )

        parts.append(
            types.Part.from_text(
                text=(
                    f"\nAUDIO FOR ITEM "
                    f"{local_index}:"
                )
            )
        )

        parts.append(
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type,
            )
        )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=parts,
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=(
                REVIEW_MAX_OUTPUT_TOKENS
            ),
            response_mime_type="application/json",
            response_schema=(
                RelabelBatchResult
                .model_json_schema()
            ),
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini review response is empty."
        )

    try:

        result = (
            RelabelBatchResult
            .model_validate_json(
                response.text
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Gemini review response could not be parsed: "
            f"{exc}"
        ) from exc

    validate_indexes(
        result,
        len(batch),
    )

    repaired = []

    for item in result.items:

        original = batch[
            item.index
        ]["transcript"]

        # Do not rewrite long transcripts.
        if is_long_transcript(
            original
        ):

            item = item.model_copy(
                update={
                    "transcript_reviewed": original
                }
            )

            repaired.append(
                item
            )

            continue

        reviewed = (
            item.transcript_reviewed
            or ""
        ).strip()

        # Noise is allowed to be empty.
        if item.intent == "BACKCHANNEL_OR_NOISE":

            if is_suspicious_generated_review(
                reviewed
            ):

                reviewed = ""

            item = item.model_copy(
                update={
                    "transcript_reviewed": reviewed
                }
            )

            repaired.append(
                item
            )

            continue

        # Normal intent.
        if not reviewed:

            reviewed = original

        elif is_suspicious_generated_review(
            reviewed
        ):

            reviewed = original

        item = item.model_copy(
            update={
                "transcript_reviewed": reviewed
            }
        )

        repaired.append(
            item
        )

    return RelabelBatchResult(
        items=repaired
    )


# ============================================================
# GEMINI INTENT ONLY
# ============================================================

def classify_intent_only_batch(
    client,
    batch: list[dict],
) -> IntentOnlyBatchResult:

    parts = [
        types.Part.from_text(
            text=build_intent_only_batch_text(
                batch
            )
        )
    ]

    for local_index, item in enumerate(
        batch
    ):

        audio_bytes, mime_type = (
            load_audio_bytes(
                item["audio"]
            )
        )

        parts.append(
            types.Part.from_text(
                text=(
                    f"\nAUDIO FOR ITEM "
                    f"{local_index}:"
                )
            )
        )

        parts.append(
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type,
            )
        )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=parts,
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=(
                INTENT_ONLY_SYSTEM_PROMPT
            ),
            temperature=0.0,
            max_output_tokens=(
                INTENT_ONLY_MAX_OUTPUT_TOKENS
            ),
            response_mime_type="application/json",
            response_schema=(
                IntentOnlyBatchResult
                .model_json_schema()
            ),
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini intent-only response is empty."
        )

    try:

        result = (
            IntentOnlyBatchResult
            .model_validate_json(
                response.text
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Gemini intent-only response could not be parsed: "
            f"{exc}"
        ) from exc

    validate_indexes(
        result,
        len(batch),
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_indexes(
    result,
    batch_size: int,
) -> None:

    expected = set(
        range(batch_size)
    )

    actual = set()

    for item in result.items:

        if item.index in actual:

            raise RuntimeError(
                f"Duplicate result index: "
                f"{item.index}"
            )

        actual.add(
            item.index
        )

    missing = (
        expected - actual
    )

    if missing:

        raise RuntimeError(
            f"Missing result indexes: "
            f"{sorted(missing)}"
        )


# ============================================================
# RETRIES
# ============================================================

def with_retries(
    fn,
    batch_size: int,
    label: str,
):

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            return fn()

        except Exception as exc:

            last_error = exc

            print(
                f"[{label}] "
                f"{batch_size} items | "
                f"attempt "
                f"{attempt}/{MAX_RETRIES} "
                f"failed: {exc}"
            )

            if attempt >= MAX_RETRIES:
                break

            delay = (
                RETRY_DELAY_S
                * (2 ** (attempt - 1))
                + random.uniform(
                    0.0,
                    0.5,
                )
            )

            print(
                f"[{label}] "
                f"retrying in {delay:.1f}s"
            )

            time.sleep(
                delay
            )

    raise last_error


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def make_fallback_result(
    batch: list[dict],
) -> RelabelBatchResult:

    items = []

    for local_index, row in enumerate(
        batch
    ):

        items.append(
            RelabelItem(
                index=local_index,
                transcript_reviewed=(
                    row["transcript"]
                ),
                intent="UNCLEAR_INPUT",
                confidence=0.0,
            )
        )

    return RelabelBatchResult(
        items=items
    )


def make_pathological_result(
    row: dict,
) -> RelabelBatchResult:

    return RelabelBatchResult(
        items=[
            RelabelItem(
                index=0,
                transcript_reviewed=(
                    row["transcript"]
                ),
                intent="UNCLEAR_INPUT",
                confidence=0.0,
            )
        ]
    )


# ============================================================
# ONE BATCH
# ============================================================

def process_one_batch(
    client,
    batch_number: int,
    batch: list[dict],
):

    """
    Failure hierarchy:

        review+intent
              ↓
        intent-only
              ↓
        deterministic UNCLEAR_INPUT

    No exception is intentionally allowed to escape.
    """

    try:

        try:

            result = with_retries(
                lambda: classify_review_batch(
                    client,
                    batch,
                ),
                len(batch),
                f"batch {batch_number} review",
            )

            return (
                batch_number,
                batch,
                result,
                "review",
            )

        except Exception as review_error:

            print(
                f"[batch {batch_number}] "
                f"review permanently failed; "
                f"trying intent-only: "
                f"{review_error}"
            )

        # ----------------------------------------------------
        # Intent-only fallback.
        # ----------------------------------------------------

        try:

            intent_result = with_retries(
                lambda: classify_intent_only_batch(
                    client,
                    batch,
                ),
                len(batch),
                f"batch {batch_number} intent-only",
            )

            by_index = {
                item.index: item
                for item in intent_result.items
            }

            output_items = []

            for local_index, row in enumerate(
                batch
            ):

                item = by_index[
                    local_index
                ]

                output_items.append(
                    RelabelItem(
                        index=local_index,
                        transcript_reviewed=(
                            row["transcript"]
                        ),
                        intent=item.intent,
                        confidence=item.confidence,
                    )
                )

            return (
                batch_number,
                batch,
                RelabelBatchResult(
                    items=output_items
                ),
                "intent-only",
            )

        except Exception as intent_error:

            print(
                f"[batch {batch_number}] "
                f"intent-only permanently failed; "
                f"using deterministic fallback: "
                f"{intent_error}"
            )

        # ----------------------------------------------------
        # Deterministic fallback.
        # ----------------------------------------------------

        return (
            batch_number,
            batch,
            make_fallback_result(
                batch
            ),
            "fallback",
        )

    except Exception as unexpected_error:

        # ----------------------------------------------------
        # LAST SAFETY NET.
        # This catches ANY unexpected Python error in the
        # batch function itself.
        # ----------------------------------------------------

        print(
            f"[batch {batch_number}] "
            f"UNEXPECTED ERROR: "
            f"{unexpected_error}"
        )

        print(
            f"[batch {batch_number}] "
            f"forcing fallback and continuing."
        )

        return (
            batch_number,
            batch,
            make_fallback_result(
                batch
            ),
            "fallback",
        )


# ============================================================
# CHECKPOINT RECORDS
# ============================================================

def make_checkpoint_records(
    batch: list[dict],
    result: RelabelBatchResult,
    method: str,
) -> list[dict]:

    by_index = {
        item.index: item
        for item in result.items
    }

    records = []

    for local_index, row in enumerate(
        batch
    ):

        item = by_index[
            local_index
        ]

        records.append(
            {
                "split": row[
                    "split"
                ],
                "dataset_index": row[
                    "dataset_index"
                ],

                # Source identifiers.
                "id": row[
                    "id"
                ],
                "oid": row[
                    "oid"
                ],
                "conversation_id": row[
                    "conversation_id"
                ],
                "recording_url": row[
                    "recording_url"
                ],
                "audio_reference": row[
                    "audio_reference"
                ],

                # Audit only.
                "original_intent": row[
                    "old_intent"
                ],
                "transcript": row[
                    "transcript"
                ],

                # Final values.
                "intent": item.intent,
                "transcript_reviewed": (
                    item.transcript_reviewed
                ),
                "confidence": item.confidence,

                "method": method,
            }
        )

    return records


# ============================================================
# CONCURRENT BATCH PROCESSING
# ============================================================

def process_batches(
    client,
    rows: list[dict],
    checkpoint_path: str,
):

    batches = [
        rows[i:i + BATCH_SIZE]
        for i in range(
            0,
            len(rows),
            BATCH_SIZE,
        )
    ]

    total_rows = len(rows)
    total_batches = len(batches)

    print(
        f"[relabel] rows to process : "
        f"{total_rows:,}"
    )

    print(
        f"[relabel] total batches   : "
        f"{total_batches:,}"
    )

    print(
        f"[relabel] batch size      : "
        f"{BATCH_SIZE}"
    )

    print(
        f"[relabel] concurrency     : "
        f"{MAX_CONCURRENCY}"
    )

    print(
        f"[relabel] checkpoint      : "
        f"{checkpoint_path}"
    )

    completed_rows = 0
    completed_batches = 0

    results = []

    method_counts = {
        "review": 0,
        "intent-only": 0,
        "fallback": 0,
    }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CONCURRENCY
    ) as executor:

        futures = {}

        for batch_number, batch in enumerate(
            batches,
            start=1,
        ):

            future = executor.submit(
                process_one_batch,
                client,
                batch_number,
                batch,
            )

            futures[
                future
            ] = (
                batch_number,
                batch,
            )

        for future in concurrent.futures.as_completed(
            futures
        ):

            batch_number, original_batch = (
                futures[future]
            )

            # ------------------------------------------------
            # THE LOOP ITSELF NEVER RAISES FOR A BATCH.
            # ------------------------------------------------

            try:

                (
                    returned_batch_number,
                    completed_batch,
                    result,
                    method,
                ) = future.result()

            except BaseException as exc:

                # Extremely defensive:
                # even an unexpected worker exception does
                # not stop other batches.
                print(
                    f"[relabel] WORKER EXCEPTION "
                    f"| batch={batch_number} "
                    f"| rows={len(original_batch)} "
                    f"| type={type(exc).__name__} "
                    f"| error={exc}"
                )

                completed_batch = (
                    original_batch
                )

                result = make_fallback_result(
                    original_batch
                )

                method = "fallback"

            # ------------------------------------------------
            # CHECKPOINT BEFORE PROGRESS.
            # ------------------------------------------------

            try:

                checkpoint_records = (
                    make_checkpoint_records(
                        completed_batch,
                        result,
                        method,
                    )
                )

                append_checkpoint(
                    checkpoint_path,
                    checkpoint_records,
                )

            except BaseException as checkpoint_error:

                # A checkpoint failure is different:
                # do NOT pretend these rows completed.
                print(
                    f"[relabel] CHECKPOINT ERROR "
                    f"| batch={batch_number} "
                    f"| error={checkpoint_error}"
                )

                # Stop because continuing without a durable
                # checkpoint can cause duplicate processing.
                raise

            method_counts[
                method
            ] += len(
                completed_batch
            )

            # ------------------------------------------------
            # In-memory results for summary only.
            # ------------------------------------------------

            try:

                by_index = {
                    item.index: item
                    for item in result.items
                }

                for local_index, row in enumerate(
                    completed_batch
                ):

                    results.append(
                        (
                            row,
                            by_index[
                                local_index
                            ],
                        )
                    )

            except Exception as summary_error:

                # Already checkpointed, so do not fail the job
                # because of summary bookkeeping.
                print(
                    f"[relabel] summary bookkeeping "
                    f"error for batch={batch_number}: "
                    f"{summary_error}"
                )

            completed_batches += 1
            completed_rows += len(
                completed_batch
            )

            percent = (
                completed_rows
                / total_rows
                * 100.0
            )

            print(
                f"[relabel] DONE "
                f"{completed_rows:,}/"
                f"{total_rows:,} "
                f"({percent:.2f}%) | "
                f"batch={batch_number}/"
                f"{total_batches} | "
                f"method={method} | "
                f"checkpointed"
            )

    print(
        "\n[relabel] METHODS:"
    )

    for method, count in method_counts.items():

        print(
            f"  {method}: "
            f"{count:,}"
        )

    return results


# ============================================================
# TEST OUTPUT
# ============================================================

def save_test_results(
    results,
    path: str,
) -> None:

    output = []

    for row, result in results:

        output.append(
            {
                "split": row[
                    "split"
                ],
                "dataset_index": row[
                    "dataset_index"
                ],
                "id": row[
                    "id"
                ],
                "original_intent": row[
                    "old_intent"
                ],
                "new_intent": result.intent,
                "transcript": row[
                    "transcript"
                ],
                "transcript_reviewed": (
                    result.transcript_reviewed
                ),
                "confidence": result.confidence,
            }
        )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# APPLY CHECKPOINT
# ============================================================

def apply_checkpoint(
    dataset_dict: DatasetDict,
    checkpoint: dict[str, dict],
) -> DatasetDict:

    updated_splits = {}

    for split in HF_DATASET_SPLITS:

        dataset = dataset_dict[
            split
        ]

        intents = list(
            dataset[
                "intent"
            ]
        )

        if (
            "transcript_reviewed"
            in dataset.column_names
        ):

            reviewed = list(
                dataset[
                    "transcript_reviewed"
                ]
            )

        else:

            reviewed = list(
                dataset[
                    "transcript"
                ]
            )

        applied = 0

        for dataset_index in range(
            len(dataset)
        ):

            key = checkpoint_key(
                split,
                dataset_index,
            )

            record = checkpoint.get(
                key
            )

            if record is None:
                continue

            intents[
                dataset_index
            ] = record[
                "intent"
            ]

            reviewed[
                dataset_index
            ] = record[
                "transcript_reviewed"
            ]

            applied += 1

        print(
            f"[relabel] checkpoint applied "
            f"to {split}: "
            f"{applied:,}"
        )

        if (
            "transcript_reviewed"
            in dataset.column_names
        ):

            dataset = dataset.remove_columns(
                [
                    "transcript_reviewed"
                ]
            )

        dataset = dataset.add_column(
            "transcript_reviewed",
            reviewed,
        )

        dataset = dataset.remove_columns(
            [
                "intent"
            ]
        )

        dataset = dataset.add_column(
            "intent",
            intents,
        )

        updated_splits[
            split
        ] = dataset

    return DatasetDict(
        updated_splits
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results,
    missing_audio_count=0,
) -> None:

    intent_changes = {}
    transcript_changes = 0

    for row, result in results:

        if (
            row["old_intent"]
            != result.intent
        ):

            key = (
                row["old_intent"],
                result.intent,
            )

            intent_changes[key] = (
                intent_changes.get(
                    key,
                    0,
                )
                + 1
            )

        if (
            row["transcript"].strip()
            != result.transcript_reviewed.strip()
        ):

            transcript_changes += 1

    print(
        "\n[relabel] ================= SUMMARY ================="
    )

    print(
        f"processed rows     : "
        f"{len(results):,}"
    )

    print(
        f"intent changes     : "
        f"{sum(intent_changes.values()):,}"
    )

    print(
        f"transcript changes : "
        f"{transcript_changes:,}"
    )

    print(
        f"missing audio      : "
        f"{missing_audio_count:,}"
    )

    if intent_changes:

        print(
            "\nIntent changes:"
        )

        for (
            (old, new),
            count,
        ) in sorted(
            intent_changes.items(),
            key=lambda x: -x[1],
        ):

            print(
                f"  {old} -> {new}: "
                f"{count:,}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Resumable multimodal intent relabeler."
        )
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--test",
        action="store_true",
        help=(
            "Process 10 uncheckpointed rows total."
        ),
    )

    mode.add_argument(
        "--full",
        action="store_true",
        help=(
            "Process all uncheckpointed rows."
        ),
    )

    mode.add_argument(
        "--build-from-checkpoint",
        action="store_true",
        help=(
            "Build HF dataset from checkpoint only."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_PATH,
    )

    parser.add_argument(
        "--missing-audio",
        default=MISSING_AUDIO_PATH,
    )

    parser.add_argument(
        "--test-output",
        default=TEST_OUTPUT,
    )

    parser.add_argument(
        "--output",
        default=FULL_OUTPUT,
    )

    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
    )

    args = parser.parse_args()

    if BATCH_SIZE != 10:
        raise RuntimeError(
            "BATCH_SIZE must be exactly 10."
        )

    # ========================================================
    # RESET
    # ========================================================

    if args.reset_checkpoint:

        for path in (
            args.checkpoint,
            args.missing_audio,
        ):

            if os.path.exists(path):

                os.remove(
                    path
                )

                print(
                    f"[relabel] deleted: "
                    f"{path}"
                )

    # ========================================================
    # LOAD DATA
    # ========================================================

    dataset_dict = load_hf_dataset()

    total_rows = sum(
        len(
            dataset_dict[split]
        )
        for split in HF_DATASET_SPLITS
    )

    print(
        f"[relabel] total dataset rows: "
        f"{total_rows:,}"
    )

    # ========================================================
    # BUILD FROM CHECKPOINT
    # ========================================================

    if args.build_from_checkpoint:

        checkpoint = load_checkpoint(
            args.checkpoint
        )

        if not checkpoint:
            raise RuntimeError(
                "Checkpoint is empty."
            )

        updated_dataset = apply_checkpoint(
            dataset_dict,
            checkpoint,
        )

        print(
            f"[relabel] saving dataset to: "
            f"{args.output}"
        )

        updated_dataset.save_to_disk(
            args.output
        )

        print(
            "[relabel] dataset built from checkpoint."
        )

        return

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = load_checkpoint(
        args.checkpoint
    )

    print(
        f"[relabel] already checkpointed: "
        f"{len(checkpoint):,}"
    )

    # ========================================================
    # SELECT
    # ========================================================

    limit = (
        10
        if args.test
        else None
    )

    (
        normal_rows,
        missing_audio,
        pathological_rows,
    ) = select_rows(
        dataset_dict,
        checkpoint,
        limit=limit,
    )

    # ========================================================
    # MISSING AUDIO
    # ========================================================

    if missing_audio:

        append_missing_audio(
            args.missing_audio,
            missing_audio,
        )

        print(
            f"[relabel] missing audio: "
            f"{len(missing_audio):,}"
        )

    # ========================================================
    # PATHOLOGICAL ROWS
    # ========================================================

    if pathological_rows:

        for row in pathological_rows:

            result = make_pathological_result(
                row
            )

            records = (
                make_checkpoint_records(
                    [row],
                    result,
                    "pathological_transcript",
                )
            )

            try:

                append_checkpoint(
                    args.checkpoint,
                    records,
                )

            except Exception as exc:

                print(
                    f"[relabel] checkpoint error "
                    f"for pathological row "
                    f"{row['split']}:{row['dataset_index']}: "
                    f"{exc}"
                )

                raise

            print(
                f"[relabel] PATHOLOGICAL "
                f"| {row['split']}:"
                f"{row['dataset_index']} "
                f"| checkpointed"
            )

    # ========================================================
    # TEST
    # ========================================================

    if args.test:

        if normal_rows:

            client = build_gemini_client()

            results = process_batches(
                client,
                normal_rows,
                args.checkpoint,
            )

            if results:
                save_test_results(
                    results,
                    args.test_output,
                )

        print(
            f"[relabel] test output: "
            f"{args.test_output}"
        )

        print(
            f"[relabel] checkpoint: "
            f"{args.checkpoint}"
        )

        print(
            "[relabel] HF dataset was NOT modified."
        )

        return

    # ========================================================
    # FULL
    # ========================================================

    print(
        "\n[relabel] ================= FULL RUN ================="
    )

    print(
        f"[relabel] checkpointed : "
        f"{len(checkpoint):,}"
    )

    print(
        f"[relabel] normal rows  : "
        f"{len(normal_rows):,}"
    )

    print(
        f"[relabel] pathological : "
        f"{len(pathological_rows):,}"
    )

    print(
        f"[relabel] missing aud : "
        f"{len(missing_audio):,}"
    )

    if normal_rows:

        client = build_gemini_client()

        process_batches(
            client,
            normal_rows,
            args.checkpoint,
        )

    else:

        print(
            "[relabel] no normal Gemini rows remain."
        )

    # ========================================================
    # FINAL DATASET FROM DURABLE CHECKPOINT
    # ========================================================

    final_checkpoint = load_checkpoint(
        args.checkpoint
    )

    print(
        f"\n[relabel] final checkpoint rows: "
        f"{len(final_checkpoint):,}"
    )

    updated_dataset = apply_checkpoint(
        dataset_dict,
        final_checkpoint,
    )

    print(
        f"[relabel] saving final dataset: "
        f"{args.output}"
    )

    updated_dataset.save_to_disk(
        args.output
    )

    print(
        "\n[relabel] COMPLETE"
    )


if __name__ == "__main__":
    main()