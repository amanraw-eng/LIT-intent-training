from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import time

from datasets import DatasetDict, load_dataset
from google import genai
from google.genai import types
from google.oauth2 import service_account

from . import config
from intent_taxonomy import INTENT_SET
from .relabel_unclear_prompt import SYSTEM_PROMPT


# ============================================================
# CONFIG
# ============================================================

SOURCE_REPO = "kapturecx/call-transcript-intent-data-v2"

SPLITS = (
    "train",
    "validation",
    "eval",
)

# Keep batches small. Output is intentionally plain text.
BATCH_SIZE = 20

MAX_CONCURRENCY = int(
    getattr(
        config,
        "RELABEL_UNCLEAR_MAX_CONCURRENCY",
        30,
    )
)

MAX_RETRIES = int(
    getattr(
        config,
        "RELABEL_UNCLEAR_MAX_RETRIES",
        3,
    )
)

RETRY_DELAY_S = float(
    getattr(
        config,
        "RELABEL_UNCLEAR_RETRY_DELAY_S",
        2.0,
    )
)

# IMPORTANT:
# Large enough to avoid premature truncation.
MAX_OUTPUT_TOKENS = 4096

CHECKPOINT_PATH = getattr(
    config,
    "RELABEL_UNCLEAR_CHECKPOINT",
    "relabel_unclear_checkpoint.jsonl",
)

TEST_OUTPUT = getattr(
    config,
    "RELABEL_UNCLEAR_TEST_OUTPUT",
    "test_unclear_relabel.json",
)

LOCAL_OUTPUT = getattr(
    config,
    "RELABEL_UNCLEAR_LOCAL_OUTPUT",
    "call-transcript-intent-data-v2-unclear-relabeled",
)

HF_TOKEN = (
    getattr(config, "HF_TOKEN", None)
    or os.getenv("HF_TOKEN")
)

GEMINI_KEY_PATH = config.GEMINI_KEY_PATH
GEMINI_LOCATION = config.GEMINI_LOCATION
GEMINI_MODEL = config.GEMINI_MODEL


# ============================================================
# CLOSED TAXONOMY
# ============================================================

INTENTS = INTENT_SET


# ============================================================
# GEMINI CLIENT
# ============================================================

def build_gemini_client():
    if not GEMINI_KEY_PATH:
        raise RuntimeError(
            "config.GEMINI_KEY_PATH is not configured."
        )

    if not os.path.exists(GEMINI_KEY_PATH):
        raise FileNotFoundError(
            f"Gemini service account file not found: "
            f"{GEMINI_KEY_PATH}"
        )

    with open(
        GEMINI_KEY_PATH,
        encoding="utf-8",
    ) as f:
        service_info = json.load(f)

    credentials = (
        service_account.Credentials.from_service_account_file(
            GEMINI_KEY_PATH,
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform"
            ],
        )
    )

    client = genai.Client(
        vertexai=True,
        project=service_info["project_id"],
        location=GEMINI_LOCATION,
        credentials=credentials,
    )

    print(
        f"[unclear] Gemini model    : {GEMINI_MODEL}"
    )
    print(
        f"[unclear] Gemini location : {GEMINI_LOCATION}"
    )

    return client


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset_v2() -> DatasetDict:
    print(
        f"[unclear] loading dataset: {SOURCE_REPO}"
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

        print(
            f"[unclear] {split}: "
            f"{len(dataset[split]):,}"
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


def load_checkpoint() -> dict[str, dict]:
    completed: dict[str, dict] = {}

    if not os.path.exists(CHECKPOINT_PATH):
        print(
            "[unclear] second-pass checkpoint not found."
        )
        return completed

    with open(
        CHECKPOINT_PATH,
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
                    int(record["dataset_index"]),
                )

                completed[key] = record

            except Exception as exc:
                raise RuntimeError(
                    f"Invalid checkpoint line "
                    f"{line_number}: {exc}"
                ) from exc

    print(
        f"[unclear] second-pass checkpointed: "
        f"{len(completed):,}"
    )

    return completed


def append_checkpoint(
    records: list[dict],
) -> None:

    if not records:
        return

    parent = os.path.dirname(
        os.path.abspath(CHECKPOINT_PATH)
    )

    if parent:
        os.makedirs(
            parent,
            exist_ok=True,
        )

    with open(
        CHECKPOINT_PATH,
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
        os.fsync(f.fileno())


# ============================================================
# SELECT ONLY UNCLEAR
# ============================================================

def select_unclear_rows(
    dataset: DatasetDict,
    checkpoint: dict[str, dict],
    limit: int | None = None,
) -> list[dict]:

    rows: list[dict] = []

    for split in SPLITS:

        ds = dataset[split]

        for dataset_index in range(
            len(ds)
        ):

            key = checkpoint_key(
                split,
                dataset_index,
            )

            # Already processed by second pass.
            if key in checkpoint:
                continue

            # Only CURRENTLY unclear rows.
            if ds[dataset_index]["intent"] != "UNCLEAR_INPUT":
                continue

            # Text only. No audio.
            text = (
                ds[dataset_index].get(
                    "transcript_reviewed"
                )
                or ds[dataset_index].get(
                    "transcript"
                )
                or ""
            ).strip()

            if not text:
                continue

            rows.append(
                {
                    "split": split,
                    "dataset_index": dataset_index,
                    "id": str(
                        ds[dataset_index].get(
                            "id",
                            dataset_index,
                        )
                    ),
                    "transcript_reviewed": text,
                }
            )

            if (
                limit is not None
                and len(rows) >= limit
            ):
                return rows

    return rows


# ============================================================
# USER PROMPT
# ============================================================

def build_user_prompt(
    batch: list[dict],
) -> str:

    count = len(batch)

    lines = [
        f"You have exactly {count} input items.",
        f"You MUST return exactly {count} intent labels.",
        "Return one label per line, in input order.",
        "Do not stop after the first item.",
        "Do not return JSON.",
        "Do not number the labels.",
        "Do not explain.",
        "",
    ]

    for index, row in enumerate(batch):

        lines.extend(
            [
                f"ITEM {index}",
                f"transcript_reviewed: "
                f"{row['transcript_reviewed']}",
                "",
            ]
        )

    lines.append(
        f"OUTPUT EXACTLY {count} LABELS:"
    )

    return "\n".join(lines)


# ============================================================
# PLAIN TEXT PARSER
# ============================================================

def parse_intents(
    response_text: str,
    expected_count: int,
) -> list[str]:
    """
    Expected response:

        INTENT
        INTENT
        INTENT

    No JSON.
    """

    text = (
        response_text
        or ""
    ).strip()

    if not text:
        raise RuntimeError(
            "Gemini returned empty response."
        )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Remove accidental markdown fences.
    lines = [
        line
        for line in lines
        if line not in {
            "```",
            "```text",
            "```plaintext",
        }
    ]

    # Tolerate accidental numbering.
    cleaned = []

    for line in lines:

        # "1. INTENT"
        parts = line.split(
            ". ",
            1,
        )

        if (
            len(parts) == 2
            and parts[0].isdigit()
        ):
            line = parts[1].strip()

        # "1) INTENT"
        parts = line.split(
            ") ",
            1,
        )

        if (
            len(parts) == 2
            and parts[0].isdigit()
        ):
            line = parts[1].strip()

        cleaned.append(
            line
        )

    lines = cleaned

    if len(lines) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} labels, "
            f"got {len(lines)}. "
            f"Raw response={response_text!r}"
        )

    for output_index, intent in enumerate(
        lines
    ):

        if intent not in INTENTS:
            raise RuntimeError(
                f"Invalid intent at output index "
                f"{output_index}: {intent!r}. "
                f"Raw response={response_text!r}"
            )

    return lines


# ============================================================
# GEMINI CALL
# ============================================================

def classify_batch(
    client,
    batch: list[dict],
) -> list[str]:

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_user_prompt(
            batch
        ),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned empty response."
        )

    return parse_intents(
        response.text,
        len(batch),
    )


# ============================================================
# RETRY
# ============================================================

def retry_call(
    fn,
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
                f"attempt {attempt}/{MAX_RETRIES} "
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

            time.sleep(delay)

    raise last_error


# ============================================================
# PROCESS ONE BATCH
# ============================================================

def process_batch(
    client,
    batch_number: int,
    batch: list[dict],
):
    """
    First:
        one 5-item plain-text request.

    If malformed/failed:
        immediately classify each item individually.

    If an individual item fails:
        keep UNCLEAR_INPUT.

    There is no recursive splitting.
    """

    # --------------------------------------------------------
    # Batch attempt.
    #
    # A malformed batch is NOT retried three times here.
    # We immediately use individual classification.
    # --------------------------------------------------------

    try:

        intents = classify_batch(
            client,
            batch,
        )

        return (
            batch_number,
            batch,
            intents,
            "batch",
        )

    except Exception as batch_error:

        print(
            f"[batch {batch_number}] "
            f"batch classification failed; "
            f"switching to individual items: "
            f"{batch_error}"
        )

    # --------------------------------------------------------
    # Individual fallback.
    # --------------------------------------------------------

    intents: list[str] = []

    for local_index, row in enumerate(
        batch
    ):

        try:

            single_intent = retry_call(
                lambda row=row: classify_batch(
                    client,
                    [row],
                ),
                (
                    f"batch {batch_number} "
                    f"item {local_index}"
                ),
            )

            if len(single_intent) != 1:
                raise RuntimeError(
                    f"Expected exactly one label, "
                    f"got {len(single_intent)}"
                )

            intents.append(
                single_intent[0]
            )

        except Exception as item_error:

            print(
                f"[batch {batch_number}] "
                f"item {local_index} failed; "
                f"keeping UNCLEAR_INPUT: "
                f"{item_error}"
            )

            intents.append(
                "UNCLEAR_INPUT"
            )

    return (
        batch_number,
        batch,
        intents,
        "individual",
    )


# ============================================================
# CHECKPOINT RECORDS
# ============================================================

def make_checkpoint_records(
    batch: list[dict],
    intents: list[str],
    method: str,
) -> list[dict]:

    if len(batch) != len(intents):
        raise RuntimeError(
            f"Cannot checkpoint: "
            f"{len(batch)} rows but "
            f"{len(intents)} intents."
        )

    records = []

    for row, intent in zip(
        batch,
        intents,
        strict=True,
    ):

        if intent not in INTENTS:
            raise RuntimeError(
                f"Invalid intent for checkpoint: "
                f"{intent!r}"
            )

        records.append(
            {
                "split": row["split"],
                "dataset_index": row[
                    "dataset_index"
                ],
                "id": row["id"],
                "old_intent": "UNCLEAR_INPUT",
                "new_intent": intent,
                "method": method,
            }
        )

    return records


# ============================================================
# APPLY SECOND PASS
# ============================================================

def apply_results(
    dataset: DatasetDict,
    checkpoint: dict[str, dict],
) -> DatasetDict:

    updated = {}

    for split in SPLITS:

        ds = dataset[split]

        intents = list(
            ds["intent"]
        )

        promoted = 0

        for dataset_index in range(
            len(ds)
        ):

            record = checkpoint.get(
                checkpoint_key(
                    split,
                    dataset_index,
                )
            )

            if record is None:
                continue

            if record.get(
                "old_intent"
            ) != "UNCLEAR_INPUT":
                continue

            # Safety: this second pass may only modify
            # rows that were UNCLEAR when this dataset
            # was loaded.
            if (
                ds[dataset_index]["intent"]
                != "UNCLEAR_INPUT"
            ):
                continue

            new_intent = record[
                "new_intent"
            ]

            if new_intent not in INTENTS:
                print(
                    f"[unclear] invalid checkpoint "
                    f"intent {new_intent!r}; "
                    f"skipping "
                    f"{split}:{dataset_index}"
                )
                continue

            intents[
                dataset_index
            ] = new_intent

            if new_intent != "UNCLEAR_INPUT":
                promoted += 1

        ds = ds.remove_columns(
            ["intent"]
        )

        ds = ds.add_column(
            "intent",
            intents,
        )

        updated[
            split
        ] = ds

        print(
            f"[unclear] {split}: "
            f"promoted {promoted:,} rows"
        )

    return DatasetDict(
        updated
    )


# ============================================================
# TEST OUTPUT
# ============================================================

def save_test_results(
    rows: list[dict],
    checkpoint: dict[str, dict],
) -> None:

    output = []

    for row in rows:

        record = checkpoint.get(
            checkpoint_key(
                row["split"],
                row["dataset_index"],
            )
        )

        if record is None:
            continue

        output.append(
            {
                "split": row["split"],
                "dataset_index": row[
                    "dataset_index"
                ],
                "id": row["id"],
                "transcript_reviewed": row[
                    "transcript_reviewed"
                ],
                "old_intent": "UNCLEAR_INPUT",
                "new_intent": record[
                    "new_intent"
                ],
                "method": record[
                    "method"
                ],
            }
        )

    with open(
        TEST_OUTPUT,
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
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Second-pass relabeling of UNCLEAR_INPUT "
            "using transcript_reviewed only."
        )
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--test",
        action="store_true",
        help="Process 10 UNCLEAR_INPUT rows.",
    )

    mode.add_argument(
        "--full",
        action="store_true",
        help="Process all remaining UNCLEAR_INPUT rows.",
    )

    parser.add_argument(
        "--push",
        action="store_true",
        help=(
            "Push updated dataset to "
            "kapturecx/call-transcript-intent-data-v2."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD
    # ========================================================

    dataset = load_dataset_v2()

    checkpoint = load_checkpoint()

    # ========================================================
    # SELECT
    # ========================================================

    rows = select_unclear_rows(
        dataset,
        checkpoint,
        limit=(
            10
            if args.test
            else None
        ),
    )

    print(
        f"[unclear] selected rows: "
        f"{len(rows):,}"
    )

    if not rows:
        print(
            "[unclear] nothing to process."
        )
        return

    # ========================================================
    # BATCHES
    # ========================================================

    batches = [
        rows[i:i + BATCH_SIZE]
        for i in range(
            0,
            len(rows),
            BATCH_SIZE,
        )
    ]

    print(
        f"[unclear] batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"[unclear] batches: "
        f"{len(batches):,}"
    )

    print(
        f"[unclear] concurrency: "
        f"{MAX_CONCURRENCY}"
    )

    print(
        f"[unclear] max output tokens: "
        f"{MAX_OUTPUT_TOKENS}"
    )

    # ========================================================
    # GEMINI
    # ========================================================

    client = build_gemini_client()

    print(
        f"[unclear] STARTED | "
        f"rows={len(rows):,} | "
        f"batches={len(batches):,} | "
        f"concurrency={MAX_CONCURRENCY}"
    )

    completed_rows = 0

    # ========================================================
    # CONCURRENT PROCESSING
    # ========================================================

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CONCURRENCY
    ) as executor:

        futures = {
            executor.submit(
                process_batch,
                client,
                batch_number,
                batch,
            ): (
                batch_number,
                batch,
            )
            for batch_number, batch
            in enumerate(
                batches,
                start=1,
            )
        }

        for future in concurrent.futures.as_completed(
            futures
        ):

            batch_number, original_batch = (
                futures[future]
            )

            # ------------------------------------------------
            # Absolute worker safety net.
            # ------------------------------------------------

            try:

                (
                    _,
                    batch,
                    intents,
                    method,
                ) = future.result()

            except BaseException as exc:

                print(
                    f"[unclear] unexpected worker exception "
                    f"| batch={batch_number} "
                    f"| error={exc}"
                )

                batch = original_batch

                intents = [
                    "UNCLEAR_INPUT"
                    for _ in batch
                ]

                method = "fallback"

            # ------------------------------------------------
            # CHECKPOINT IMMEDIATELY.
            # ------------------------------------------------

            try:

                records = (
                    make_checkpoint_records(
                        batch,
                        intents,
                        method,
                    )
                )

                append_checkpoint(
                    records
                )

            except BaseException as exc:

                print(
                    f"[unclear] CHECKPOINT ERROR "
                    f"| batch={batch_number} "
                    f"| error={exc}"
                )

                # Do not pretend a row is completed if its
                # result was not durably written.
                raise

            completed_rows += len(
                batch
            )

            promoted = sum(
                intent != "UNCLEAR_INPUT"
                for intent in intents
            )

            print(
                f"[unclear] DONE "
                f"{completed_rows:,}/"
                f"{len(rows):,} "
                f"({completed_rows / len(rows) * 100:.2f}%) "
                f"| batch={batch_number} "
                f"| method={method} "
                f"| promoted={promoted} "
                f"| checkpointed"
            )

    # ========================================================
    # TEST
    # ========================================================

    if args.test:

        final_checkpoint = load_checkpoint()

        save_test_results(
            rows,
            final_checkpoint,
        )

        print(
            f"[unclear] test results: "
            f"{TEST_OUTPUT}"
        )

        print(
            "[unclear] dataset NOT modified."
        )

        return

    # ========================================================
    # FULL
    # ========================================================

    final_checkpoint = load_checkpoint()

    updated = apply_results(
        dataset,
        final_checkpoint,
    )

    print(
        f"[unclear] saving local dataset: "
        f"{LOCAL_OUTPUT}"
    )

    updated.save_to_disk(
        LOCAL_OUTPUT
    )

    print(
        "[unclear] local dataset saved."
    )

    # ========================================================
    # OPTIONAL PUSH
    # ========================================================

    if args.push:

        print(
            f"[unclear] pushing to: "
            f"{SOURCE_REPO}"
        )

        updated.push_to_hub(
            SOURCE_REPO,
            token=HF_TOKEN,
        )

        print(
            "[unclear] PUSH COMPLETE"
        )

    else:

        print(
            "[unclear] --push not specified; "
            "HF dataset NOT modified."
        )


if __name__ == "__main__":
    main()