from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from shared_config import PROJECT_ROOT, load_environment, project_path, section
from google import genai
from google.genai import types
from google.oauth2 import service_account


# ============================================================
# COMMON SETTINGS (edit shared_settings.py)
# ============================================================
load_environment(legacy_env=Path(__file__).resolve().parent / ".env")
_AUGMENTED17 = section("pipeline")["augmented17"]
DATA_DIR = PROJECT_ROOT / _AUGMENTED17["data_dir"]
INPUT_JSONL = DATA_DIR / _AUGMENTED17["input_jsonl"]
PATH_UPDATED_JSONL = DATA_DIR / _AUGMENTED17["path_updated_jsonl"]
FINAL_JSONL = DATA_DIR / _AUGMENTED17["final_jsonl"]
CHECKPOINT_JSONL = DATA_DIR / _AUGMENTED17["checkpoint_jsonl"]
ERROR_JSONL = DATA_DIR / _AUGMENTED17["error_jsonl"]
AUDIO_DIR = DATA_DIR / "audio"


# ============================================================
# PATH REWRITE
# ============================================================

OLD_AUDIO_PREFIX = _AUGMENTED17["old_audio_prefix"]
NEW_AUDIO_PREFIX = _AUGMENTED17["new_audio_prefix"] or str(AUDIO_DIR)


# ============================================================
# GEMINI
# ============================================================

GEMINI_MODEL = section("pipeline")["gemini_model"]

GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", section("pipeline")["gemini_location"])

# ============================================================
# SCRIPT-SPECIFIC RUN OVERRIDES (edit here for this migration)
# ============================================================
BATCH_SIZE = 20
MAX_CONCURRENCY = 30
MAX_OUTPUT_TOKENS = 4096
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0


# ============================================================
# SECRETS
# ============================================================

GEMINI_KEY_PATH = project_path(os.getenv("GEMINI_KEY_PATH", ".secrets/gemini-service-account.json"))


# ============================================================
# EXACT 17-INTENT TAXONOMY
# ============================================================

INTENTS = (
    "AFFIRMATIVE_ACKNOWLEDGEMENT",
    "BACKCHANNEL_OR_NOISE",
    "CALL_DEFER",
    "CONTINUE_CONVERSATION",
    "DO_NOT_CALL",
    "END_CALL",
    "IDENTITY_CONFIRMED",
    "NEGATIVE_ACKNOWLEDGEMENT",
    "NO_PAYMENT_REASON",
    "PAID_ALREADY",
    "PAY_LATER_AGREE",
    "PAY_NOW_AGREE",
    "REFUSE_TO_PAY",
    "THIRD_PARTY_AVAILABLE",
    "THIRD_PARTY_UNAVAILABLE",
    "UNCLEAR_INPUT",
    "WRONG_NUMBER",
)

INTENT_SET = set(
    INTENTS
)


# ============================================================
# GEMINI PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are a HIGH-PRECISION intent relabeler for Hindi/Hinglish borrower
utterances from Indian loan / EMI / payment collection calls.

The existing intent field may contain OLD taxonomy labels.
IGNORE the existing intent field completely.

Classify the transcript using ONLY the current utterance and the current
17-intent taxonomy below.

Use semantic meaning and communicative function, not exact keyword matching.

RULES:
- Do not trust the old intent.
- Do not infer missing context.
- Do not use previous conversation turns.
- If multiple intents are reasonably plausible, use UNCLEAR_INPUT.
- False positives are worse than UNCLEAR_INPUT.
- Be conservative.
- Return exactly one label for each input.
- Never invent an intent.

CURRENT 17 INTENTS:

AFFIRMATIVE_ACKNOWLEDGEMENT
Generic positive acknowledgement/acceptance with no more specific meaning.

BACKCHANNEL_OR_NOISE
Clearly meaningless/non-semantic vocalization/noise that can safely be
ignored. Meaningful "हाँ", "जी", "हाँ हाँ", "जी जी" should normally NOT
be this.

CALL_DEFER
Customer clearly asks for the caller/contact to happen later.

CONTINUE_CONVERSATION
Customer clearly prompts, permits, or asks the agent to continue speaking,
explaining, telling, or proceeding.

This is semantic rather than exact phrase matching.

Examples in spirit:
"हाँ जी, बताइए"
"जी मैम, बोलिए"
"आगे बताइए"
"और बताओ"
"अच्छा कहिए"

Generic acknowledgement is NOT enough:
"हाँ" -> AFFIRMATIVE_ACKNOWLEDGEMENT
"जी" -> AFFIRMATIVE_ACKNOWLEDGEMENT
"ठीक है" -> AFFIRMATIVE_ACKNOWLEDGEMENT

DO_NOT_CALL
Customer clearly says not to call again / not to call this number.

END_CALL
Clear goodbye or explicit ending of the call.

IDENTITY_CONFIRMED
Customer clearly establishes that they are the intended person.

NEGATIVE_ACKNOWLEDGEMENT
Generic negative acknowledgement without a more specific meaning.

NO_PAYMENT_REASON
Customer clearly gives a concrete reason for non-payment/payment delay.

PAID_ALREADY
Customer clearly says payment was already completed.

PAY_LATER_AGREE
Customer clearly commits to paying later/future.

PAY_NOW_AGREE
Customer clearly commits to paying now/immediately.

REFUSE_TO_PAY
Customer clearly refuses or is unwilling to pay.
Do not confuse refusal with inability.

THIRD_PARTY_AVAILABLE
Another person is clearly available, being brought to the phone, or
being handed the phone.

THIRD_PARTY_UNAVAILABLE
The intended person is clearly unavailable/cannot speak now.

UNCLEAR_INPUT
Safe default for vague, ambiguous, incomplete, context-dependent,
or uncertain utterances.

WRONG_NUMBER
Customer clearly indicates the caller has reached the wrong number,
wrong person, or wrong contact.

IMPORTANT EXAMPLES:

"हाँ जी, बताइए।"
-> CONTINUE_CONVERSATION

"जी मैम, सही समय बोलिए।"
-> CONTINUE_CONVERSATION

"हाँ"
-> AFFIRMATIVE_ACKNOWLEDGEMENT

"जी"
-> AFFIRMATIVE_ACKNOWLEDGEMENT

"मैं ही हूँ"
-> IDENTITY_CONFIRMED

"गलत नंबर है"
-> WRONG_NUMBER

"आप गलत नंबर पर फोन कर रहे हैं"
-> WRONG_NUMBER

"अब कॉल मत करना"
-> DO_NOT_CALL

"बाद में कॉल करना"
-> CALL_DEFER

"अभी कर देता हूँ"
-> PAY_NOW_AGREE

"कल कर दूंगा"
-> PAY_LATER_AGREE

"पेमेंट कर दी है"
-> PAID_ALREADY

"पैसे नहीं हैं"
-> NO_PAYMENT_REASON

"नहीं दूंगा"
-> REFUSE_TO_PAY

"क्या है?"
-> UNCLEAR_INPUT

"कौन सा है?"
-> UNCLEAR_INPUT

OUTPUT:
Return exactly one intent label per line.
Return labels in the same order as the inputs.
Do not return JSON.
Do not number the labels.
Do not add explanations.
Do not add markdown.
"""


# ============================================================
# THREAD-LOCAL GEMINI CLIENT
# ============================================================

_thread_local = threading.local()


def get_gemini_client():
    client = getattr(
        _thread_local,
        "client",
        None,
    )

    if client is not None:
        return client

    if not GEMINI_KEY_PATH:
        raise RuntimeError(
            "GEMINI_KEY_PATH is not set."
        )

    key_path = Path(
        GEMINI_KEY_PATH
    )

    if not key_path.exists():
        raise FileNotFoundError(
            f"GEMINI_KEY_PATH not found: "
            f"{key_path}"
        )

    with open(
        key_path,
        encoding="utf-8",
    ) as f:
        service_info = json.load(
            f
        )

    credentials = (
        service_account.Credentials
        .from_service_account_file(
            str(key_path),
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform"
            ],
        )
    )

    client = genai.Client(
        vertexai=True,
        project=service_info[
            "project_id"
        ],
        location=GEMINI_LOCATION,
        credentials=credentials,
    )

    _thread_local.client = client

    return client


# ============================================================
# STEP 1
# ============================================================

def rewrite_paths():
    """
    STEP 1 ONLY:

    Rewrite the old audio path prefix to the current machine path.

    No audio checking.
    No renaming.
    No Gemini.
    No intent changes.

    Input:
        data.jsonl

    Output:
        path_updated_augmented_data17.jsonl
    """

    if not INPUT_JSONL.exists():
        raise FileNotFoundError(
            f"Input JSONL not found: "
            f"{INPUT_JSONL}"
        )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "============================================================"
    )
    print(
        "[step1] PATH REWRITE"
    )
    print(
        "============================================================"
    )

    print(
        f"[step1] input : {INPUT_JSONL}"
    )

    print(
        f"[step1] output: {PATH_UPDATED_JSONL}"
    )

    print(
        f"[step1] old prefix:"
    )

    print(
        f"  {OLD_AUDIO_PREFIX}"
    )

    print(
        f"[step1] new prefix:"
    )

    print(
        f"  {NEW_AUDIO_PREFIX}"
    )

    total = 0
    changed = 0

    with open(
        INPUT_JSONL,
        "r",
        encoding="utf-8",
    ) as src, open(
        PATH_UPDATED_JSONL,
        "w",
        encoding="utf-8",
    ) as dst:

        for line_number, line in enumerate(
            src,
            start=1,
        ):

            if not line.strip():
                continue

            try:
                row = json.loads(
                    line
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

            total += 1

            old_path = str(
                row.get(
                    "chunk_path",
                    "",
                )
            )

            if old_path.startswith(
                OLD_AUDIO_PREFIX
            ):

                suffix = old_path[
                    len(
                        OLD_AUDIO_PREFIX
                    ):
                ]

                # Ensure exactly one separator.
                new_path = (
                    NEW_AUDIO_PREFIX
                    + suffix
                )

                row["chunk_path"] = (
                    new_path
                )

                changed += 1

            dst.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"[step1] rows       : {total:,}"
    )

    print(
        f"[step1] paths updated: {changed:,}"
    )

    print(
        "[step1] COMPLETE"
    )

    print(
        f"[step1] created: "
        f"{PATH_UPDATED_JSONL}"
    )


# ============================================================
# JSONL LOAD
# ============================================================

def load_jsonl(
    path: Path,
):
    rows = []

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

                row = json.loads(
                    line
                )

            except Exception as exc:

                raise RuntimeError(
                    f"Invalid JSON at line "
                    f"{line_number}: "
                    f"{exc}"
                ) from exc

            rows.append(
                row
            )

    return rows


# ============================================================
# AUDIO CHECK FOR STEP 2
# ============================================================

def validate_audio_paths(
    rows,
):
    """
    Now that STEP 1 has rewritten the paths, verify that those
    paths actually exist on the new machine.
    """

    missing = []

    for index, row in enumerate(
        rows
    ):

        path = Path(
            str(
                row.get(
                    "chunk_path",
                    "",
                )
            )
        )

        if not path.exists():
            missing.append(
                {
                    "source_index": index,
                    "chunk_path": str(
                        path
                    ),
                    "oid": row.get(
                        "oid"
                    ),
                }
            )

    return missing


# ============================================================
# PARSE GEMINI LABELS
# ============================================================

def parse_labels(
    response_text: str,
    expected_count: int,
):
    if not response_text:
        raise RuntimeError(
            "Gemini returned empty response."
        )

    lines = [
        line.strip()
        for line in response_text.splitlines()
        if line.strip()
    ]

    # Remove accidental fences.
    lines = [
        line
        for line in lines
        if line not in {
            "```",
            "```text",
            "```plaintext",
        }
    ]

    # Remove accidental numbering.
    cleaned = []

    for line in lines:

        pieces = line.split(
            ". ",
            1,
        )

        if (
            len(pieces) == 2
            and pieces[0].isdigit()
        ):
            line = pieces[1].strip()

        pieces = line.split(
            ") ",
            1,
        )

        if (
            len(pieces) == 2
            and pieces[0].isdigit()
        ):
            line = pieces[1].strip()

        cleaned.append(
            line
        )

    lines = cleaned

    if len(lines) != expected_count:

        raise RuntimeError(
            f"Expected {expected_count} "
            f"labels, got {len(lines)}. "
            f"Raw response={response_text!r}"
        )

    for index, label in enumerate(
        lines
    ):

        if label not in INTENT_SET:

            raise RuntimeError(
                f"Invalid intent at output "
                f"index {index}: "
                f"{label!r}. "
                f"Raw response={response_text!r}"
            )

    return lines


# ============================================================
# GEMINI BATCH
# ============================================================

def classify_batch(
    rows,
):
    """
    Only transcript is sent.
    Existing intent is deliberately included nowhere in the prompt.
    """

    lines = [
        (
            f"You have exactly {len(rows)} items."
        ),
        (
            f"Return exactly {len(rows)} "
            "intent labels."
        ),
        "One label per line.",
        "Same order as the inputs.",
        "Do not return JSON.",
        "Do not stop early.",
        "",
    ]

    for index, row in enumerate(
        rows
    ):

        lines.extend(
            [
                f"ITEM {index}",
                (
                    "transcript: "
                    f"{row.get('transcript', '')}"
                ),
                "",
            ]
        )

    prompt = "\n".join(
        lines
    )

    client = get_gemini_client()

    response = (
        client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    SYSTEM_PROMPT
                ),
                temperature=0.0,
                max_output_tokens=(
                    MAX_OUTPUT_TOKENS
                ),
            ),
        )
    )

    return parse_labels(
        response.text,
        len(rows),
    )


# ============================================================
# RETRY
# ============================================================

def classify_with_retries(
    rows,
    label,
):
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            return classify_batch(
                rows
            )

        except Exception as exc:

            last_error = exc

            print(
                f"[{label}] "
                f"attempt {attempt}/"
                f"{MAX_RETRIES} failed: "
                f"{exc}"
            )

            if (
                attempt
                < MAX_RETRIES
            ):

                delay = (
                    RETRY_DELAY_SECONDS
                    * attempt
                    + random.uniform(
                        0,
                        0.5,
                    )
                )

                print(
                    f"[{label}] "
                    f"retrying in "
                    f"{delay:.1f}s"
                )

                time.sleep(
                    delay
                )

    raise last_error


# ============================================================
# PROCESS BATCH
# ============================================================

def process_batch(
    batch_number,
    rows,
):
    """
    Try a batch of 20.

    If the batch fails:
        immediately process each item separately.

    If an individual classification fails:
        retain the ORIGINAL intent only if it is one of the current
        17 intents. Otherwise use UNCLEAR_INPUT.

    No batch failure is allowed to stop the migration.
    """

    print(
        f"[batch {batch_number}] START "
        f"| items={len(rows)} "
        f"| first_source_index="
        f"{rows[0]['_source_index']}"
    )

    try:

        labels = classify_with_retries(
            rows,
            f"batch {batch_number}",
        )

        print(
            f"[batch {batch_number}] "
            f"BATCH OK"
        )

        return (
            batch_number,
            rows,
            labels,
            "batch",
        )

    except Exception as exc:

        print(
            f"[batch {batch_number}] "
            f"BATCH FAILED -> "
            f"individual fallback: "
            f"{exc}"
        )

    labels = []

    for local_index, row in enumerate(
        rows
    ):

        try:

            result = classify_with_retries(
                [row],
                (
                    f"batch {batch_number} "
                    f"item {local_index + 1}"
                ),
            )

            if len(result) != 1:
                raise RuntimeError(
                    f"Expected 1 label, "
                    f"got {len(result)}"
                )

            labels.append(
                result[0]
            )

        except Exception as exc:

            print(
                f"[batch {batch_number}] "
                f"item {local_index + 1} "
                f"FAILED -> fallback: "
                f"{exc}"
            )

            old_intent = str(
                row.get(
                    "intent",
                    "",
                )
            )

            if old_intent in INTENT_SET:
                labels.append(
                    old_intent
                )
            else:
                labels.append(
                    "UNCLEAR_INPUT"
                )

    return (
        batch_number,
        rows,
        labels,
        "individual",
    )


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint():
    completed = {}

    if not CHECKPOINT_JSONL.exists():
        return completed

    with open(
        CHECKPOINT_JSONL,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

                completed[
                    int(
                        record[
                            "_source_index"
                        ]
                    )
                ] = record

            except Exception:
                # Ignore malformed checkpoint entries.
                continue

    return completed


def append_checkpoint(
    record,
):
    with open(
        CHECKPOINT_JSONL,
        "a",
        encoding="utf-8",
    ) as f:

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


def append_error(
    record,
):
    with open(
        ERROR_JSONL,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()


# ============================================================
# STEP 2: RELABEL + RENAME
# ============================================================

def relabel_and_rename():
    """
    STEP 2:

    1. Read path-updated JSONL.
    2. Check CURRENT paths.
    3. Send transcript to Gemini.
    4. Relabel to current 17-intent taxonomy.
    5. Rename actual audio.
    6. Write final JSONL.

    Audio stays in the same AUDIO_DIR.
    Only the filename changes.
    """

    if not PATH_UPDATED_JSONL.exists():

        raise FileNotFoundError(
            f"STEP 1 output not found: "
            f"{PATH_UPDATED_JSONL}\n"
            "Run --rewrite-paths first."
        )

    if not AUDIO_DIR.exists():

        raise FileNotFoundError(
            f"Audio directory not found: "
            f"{AUDIO_DIR}"
        )

    print(
        "============================================================"
    )
    print(
        "[step2] RELABEL + RENAME"
    )
    print(
        "============================================================"
    )

    print(
        f"[step2] input : {PATH_UPDATED_JSONL}"
    )

    print(
        f"[step2] output: {FINAL_JSONL}"
    )

    print(
        f"[step2] audio : {AUDIO_DIR}"
    )

    print(
        f"[step2] Gemini: {GEMINI_MODEL}"
    )

    print(
        f"[step2] batch : {BATCH_SIZE}"
    )

    print(
        f"[step2] workers: {MAX_CONCURRENCY}"
    )

    print(
        f"[step2] max tokens: "
        f"{MAX_OUTPUT_TOKENS}"
    )

    rows = load_jsonl(
        PATH_UPDATED_JSONL
    )

    print(
        f"[step2] total rows: "
        f"{len(rows):,}"
    )

    # --------------------------------------------------------
    # Add stable source index.
    # --------------------------------------------------------

    for index, row in enumerate(
        rows
    ):
        row["_source_index"] = index

    # --------------------------------------------------------
    # Existing checkpoint.
    # --------------------------------------------------------

    checkpoint = load_checkpoint()

    print(
        f"[step2] checkpointed: "
        f"{len(checkpoint):,}"
    )

    pending = [
        row
        for row in rows
        if row["_source_index"]
        not in checkpoint
    ]

    print(
        f"[step2] pending: "
        f"{len(pending):,}"
    )

    if not pending:

        print(
            "[step2] everything already processed."
        )

        return

    # --------------------------------------------------------
    # NOW, AND ONLY NOW, CHECK AUDIO PATHS.
    # --------------------------------------------------------

    missing = validate_audio_paths(
        pending
    )

    if missing:

        print(
            f"[step2] missing audio: "
            f"{len(missing):,}"
        )

        for record in missing[:20]:

            append_error(
                {
                    "type": "missing_audio",
                    **record,
                }
            )

        print(
            "[step2] Missing-audio rows will be skipped."
        )

    ready = [
        row
        for row in pending
        if Path(
            str(
                row.get(
                    "chunk_path",
                    "",
                )
            )
        ).exists()
    ]

    print(
        f"[step2] audio-ready rows: "
        f"{len(ready):,}"
    )

    if not ready:

        print(
            "[step2] no rows with usable audio."
        )

        return

    # --------------------------------------------------------
    # Build batches.
    # --------------------------------------------------------

    batches = [
        ready[i:i + BATCH_SIZE]
        for i in range(
            0,
            len(ready),
            BATCH_SIZE,
        )
    ]

    print(
        f"[step2] batches: "
        f"{len(batches):,}"
    )

    # --------------------------------------------------------
    # Classify concurrently.
    # --------------------------------------------------------

    classification_results = {}

    completed = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CONCURRENCY
    ) as executor:

        futures = {
            executor.submit(
                process_batch,
                batch_number,
                batch,
            ): batch_number
            for batch_number, batch
            in enumerate(
                batches,
                start=1,
            )
        }

        for future in concurrent.futures.as_completed(
            futures
        ):

            batch_number = futures[
                future
            ]

            try:

                (
                    _,
                    batch,
                    labels,
                    method,
                ) = future.result()

            except BaseException as exc:

                # Absolute safety net.
                batch = batches[
                    batch_number - 1
                ]

                print(
                    f"[step2] unexpected worker "
                    f"failure in batch "
                    f"{batch_number}: "
                    f"{exc}"
                )

                labels = [
                    (
                        row.get(
                            "intent"
                        )
                        if row.get(
                            "intent"
                        )
                        in INTENT_SET
                        else "UNCLEAR_INPUT"
                    )
                    for row in batch
                ]

                method = (
                    "worker_fallback"
                )

                append_error(
                    {
                        "type": (
                            "unexpected_worker_error"
                        ),
                        "batch_number": (
                            batch_number
                        ),
                        "error": str(
                            exc
                        ),
                    }
                )

            for row, label in zip(
                batch,
                labels,
            ):

                classification_results[
                    row["_source_index"]
                ] = {
                    "intent": label,
                    "method": method,
                }

            completed += len(
                batch
            )

            promoted = sum(
                1
                for label in labels
                if label != "UNCLEAR_INPUT"
            )

            print(
                f"[step2] CLASSIFIED "
                f"{completed:,}/"
                f"{len(ready):,} "
                f"({100 * completed / len(ready):.2f}%) "
                f"| batch={batch_number} "
                f"| method={method} "
                f"| non-unclear={promoted}"
            )

    # --------------------------------------------------------
    # Create/append final output.
    #
    # Each completed row is written in source order.
    # For resumability, we use the checkpoint to avoid repeating
    # Gemini and audio work.
    # --------------------------------------------------------

    print(
        "\n[step2] applying labels and renaming audio..."
    )

    # Determine next incremental filename ID.
    #
    # We always allocate IDs from the final-output/checkpoint
    # state so they remain stable during a resume.
    used_ids = set()

    if FINAL_JSONL.exists():

        try:

            existing_final = load_jsonl(
                FINAL_JSONL
            )

            for row in existing_final:

                path = Path(
                    str(
                        row.get(
                            "chunk_path",
                            "",
                        )
                    )
                )

                stem = path.stem

                # Filename pattern:
                # INTENT_00000001
                if "_" in stem:

                    suffix = stem.rsplit(
                        "_",
                        1,
                    )[-1]

                    if suffix.isdigit():
                        used_ids.add(
                            int(
                                suffix
                            )
                        )

        except Exception as exc:

            print(
                f"[step2] warning: could not "
                f"read existing final IDs: "
                f"{exc}"
            )

    next_id = (
        max(
            used_ids,
            default=0,
        )
        + 1
    )

    # --------------------------------------------------------
    # Open output once.
    # --------------------------------------------------------

    with open(
        FINAL_JSONL,
        "a",
        encoding="utf-8",
    ) as output_f:

        for row in rows:

            source_index = (
                row[
                    "_source_index"
                ]
            )

            # ------------------------------------------------
            # Already checkpointed:
            # don't touch again.
            # ------------------------------------------------

            if source_index in checkpoint:
                continue

            # ------------------------------------------------
            # No usable audio:
            # don't write a training row whose audio is absent.
            # ------------------------------------------------

            current_audio = Path(
                str(
                    row.get(
                        "chunk_path",
                        "",
                    )
                )
            )

            if not current_audio.exists():

                continue

            # ------------------------------------------------
            # Need classification.
            # ------------------------------------------------

            result = classification_results.get(
                source_index
            )

            if result is None:

                print(
                    f"[step2] no classification result "
                    f"for source index "
                    f"{source_index}; "
                    f"keeping original current intent"
                )

                new_intent = (
                    row.get(
                        "intent"
                    )
                    if row.get(
                        "intent"
                    )
                    in INTENT_SET
                    else "UNCLEAR_INPUT"
                )

                method = (
                    "missing_result_fallback"
                )

            else:

                new_intent = result[
                    "intent"
                ]

                method = result[
                    "method"
                ]

            # ------------------------------------------------
            # Rename in same audio directory.
            # ------------------------------------------------

            suffix = (
                current_audio.suffix
                or ".wav"
            )

            new_filename = (
                f"{new_intent}_"
                f"{next_id:08d}"
                f"{suffix}"
            )

            new_audio_path = (
                AUDIO_DIR
                / new_filename
            )

            # Collision protection.
            while new_audio_path.exists():

                next_id += 1

                new_filename = (
                    f"{new_intent}_"
                    f"{next_id:08d}"
                    f"{suffix}"
                )

                new_audio_path = (
                    AUDIO_DIR
                    / new_filename
                )

            try:

                # Move within the same audio directory.
                current_audio.rename(
                    new_audio_path
                )

            except Exception as exc:

                print(
                    f"[step2] AUDIO RENAME FAILED "
                    f"| source_index={source_index} "
                    f"| {current_audio} -> "
                    f"{new_audio_path} "
                    f"| {exc}"
                )

                append_error(
                    {
                        "type": "rename_failed",
                        "source_index": (
                            source_index
                        ),
                        "old_path": str(
                            current_audio
                        ),
                        "new_path": str(
                            new_audio_path
                        ),
                        "intent": (
                            new_intent
                        ),
                        "error": str(
                            exc
                        ),
                    }
                )

                continue

            # ------------------------------------------------
            # Build final row.
            # ------------------------------------------------

            output_row = dict(
                row
            )

            output_row.pop(
                "_source_index",
                None,
            )

            output_row[
                "intent"
            ] = new_intent

            output_row[
                "chunk_path"
            ] = str(
                new_audio_path
                .resolve()
            )

            # ------------------------------------------------
            # Write final JSONL.
            # ------------------------------------------------

            output_f.write(
                json.dumps(
                    output_row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_f.flush()

            os.fsync(
                output_f.fileno()
            )

            # ------------------------------------------------
            # Checkpoint AFTER both rename + JSONL write.
            # ------------------------------------------------

            checkpoint_record = {
                "_source_index": (
                    source_index
                ),
                "oid": output_row.get(
                    "oid"
                ),
                "old_intent": row.get(
                    "intent"
                ),
                "new_intent": (
                    new_intent
                ),
                "method": method,
                "old_chunk_path": (
                    str(
                        current_audio
                    )
                ),
                "new_chunk_path": (
                    str(
                        new_audio_path.resolve()
                    )
                ),
            }

            append_checkpoint(
                checkpoint_record
            )

            next_id += 1

            if (
                next_id
                % 100
                == 0
            ):

                print(
                    f"[step2] renamed/written "
                    f"through ID "
                    f"{next_id - 1:,}"
                )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    final_counts = {}

    if FINAL_JSONL.exists():

        try:

            final_rows = load_jsonl(
                FINAL_JSONL
            )

            for row in final_rows:

                intent = row.get(
                    "intent",
                    "NO_INTENT",
                )

                final_counts[
                    intent
                ] = (
                    final_counts.get(
                        intent,
                        0,
                    )
                    + 1
                )

        except Exception as exc:

            print(
                f"[step2] warning: could not "
                f"read final JSONL: "
                f"{exc}"
            )

    print(
        "\n"
        "============================================================"
    )

    print(
        "[step2] COMPLETE"
    )

    print(
        "============================================================"
    )

    print(
        f"Final JSONL: "
        f"{FINAL_JSONL}"
    )

    print(
        f"Audio dir: "
        f"{AUDIO_DIR}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_JSONL}"
    )

    print(
        f"Errors: "
        f"{ERROR_JSONL}"
    )

    print(
        "\nFinal intent distribution:"
    )

    for intent in INTENTS:

        print(
            f"  {intent:40s} "
            f"{final_counts.get(intent, 0):8,d}"
        )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Two-step migration of imported augmented data "
            "to the current 17-intent taxonomy."
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    sub.add_parser(
        "rewrite-paths",
        help=(
            "ONLY rewrite old chunk_path prefix. "
            "Does not check audio or run Gemini."
        ),
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    sub.add_parser(
        "relabel",
        help=(
            "Relabel using Gemini and rename actual audio."
        ),
    )

    # --------------------------------------------------------
    # Both
    # --------------------------------------------------------

    sub.add_parser(
        "all",
        help=(
            "Run rewrite-paths and then relabel."
        ),
    )

    args = parser.parse_args()

    if args.command == "rewrite-paths":

        rewrite_paths()

    elif args.command == "relabel":

        relabel_and_rename()

    elif args.command == "all":

        rewrite_paths()
        relabel_and_rename()


if __name__ == "__main__":
    main()