from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any

from shared_config import PROJECT_ROOT, load_environment, project_path, section
from google import genai
from google.genai import types
from google.oauth2 import service_account
from pydantic import BaseModel, Field

# ============================================================
# COMMON SETTINGS (edit shared_settings.py)
# ============================================================
load_environment(legacy_env=Path(__file__).resolve().parent / ".env")
_AUGMENTED17 = section("pipeline")["augmented17"]
DATA_DIR = PROJECT_ROOT / _AUGMENTED17["data_dir"]
INPUT_JSONL = DATA_DIR / _AUGMENTED17["input_jsonl"]
PATH_UPDATED_JSONL = INPUT_JSONL
FINAL_JSONL = DATA_DIR / _AUGMENTED17["final_jsonl"]
CHECKPOINT_JSONL = DATA_DIR / _AUGMENTED17["checkpoint_jsonl"]
ERROR_JSONL = DATA_DIR / _AUGMENTED17["error_jsonl"]
AUDIO_DIR = DATA_DIR / "audio"

# ============================================================
# PATH REWRITE PREFIXES
# ============================================================
OLD_AUDIO_PREFIX = _AUGMENTED17["old_audio_prefix"]
NEW_AUDIO_PREFIX = _AUGMENTED17["new_audio_prefix"] or str(AUDIO_DIR)

## ============================================================
# TUNED OPTIMIZATIONS
# ============================================================
GEMINI_MODEL = section("pipeline")["gemini_model"]
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", section("pipeline")["gemini_location"])
# ============================================================
# SCRIPT-SPECIFIC RUN OVERRIDES (edit here for this relabel run)
# ============================================================
BATCH_SIZE = 100
MAX_CONCURRENCY = 10
MAX_OUTPUT_TOKENS = 8192
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0
GEMINI_KEY_PATH = project_path(os.getenv("GEMINI_KEY_PATH", ".secrets/gemini-service-account.json"))

# ============================================================
# PYDANTIC STRICT 17-INTENT SCHEMA
# ============================================================
class IntentEnum(str, Enum):
    AFFIRMATIVE_ACKNOWLEDGEMENT = "AFFIRMATIVE_ACKNOWLEDGEMENT"
    BACKCHANNEL_OR_NOISE = "BACKCHANNEL_OR_NOISE"
    CALL_DEFER = "CALL_DEFER"
    CONTINUE_CONVERSATION = "CONTINUE_CONVERSATION"
    DO_NOT_CALL = "DO_NOT_CALL"
    END_CALL = "END_CALL"
    IDENTITY_CONFIRMED = "IDENTITY_CONFIRMED"
    NEGATIVE_ACKNOWLEDGEMENT = "NEGATIVE_ACKNOWLEDGEMENT"
    NO_PAYMENT_REASON = "NO_PAYMENT_REASON"
    PAID_ALREADY = "PAID_ALREADY"
    PAY_LATER_AGREE = "PAY_LATER_AGREE"
    PAY_NOW_AGREE = "PAY_NOW_AGREE"
    REFUSE_TO_PAY = "REFUSE_TO_PAY"
    THIRD_PARTY_AVAILABLE = "THIRD_PARTY_AVAILABLE"
    THIRD_PARTY_UNAVAILABLE = "THIRD_PARTY_UNAVAILABLE"
    UNCLEAR_INPUT = "UNCLEAR_INPUT"
    WRONG_NUMBER = "WRONG_NUMBER"

class SingleItemResult(BaseModel):
    item_index: int = Field(description="0-based index of item in batch")
    intent: IntentEnum = Field(description="Classified intent from the 17 allowed intents")

class BatchClassificationResponse(BaseModel):
    results: list[SingleItemResult]

INTENT_SET = {e.value for e in IntentEnum}

SYSTEM_PROMPT = r"""
You are a HIGH-PRECISION intent relabeler for Hindi/Hinglish borrower
utterances from Indian loan / EMI / payment collection calls.

Ignore the old intent. Classify using ONLY the current utterance into one of 17 intents.
Return strictly valid JSON.
"""

_thread_local = threading.local()

def get_gemini_client():
    client = getattr(_thread_local, "client", None)
    if client is not None:
        return client

    if not GEMINI_KEY_PATH:
        raise RuntimeError("GEMINI_KEY_PATH is not set.")

    key_path = Path(GEMINI_KEY_PATH)
    with open(key_path, encoding="utf-8") as f:
        service_info = json.load(f)

    credentials = service_account.Credentials.from_service_account_file(
        str(key_path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    client = genai.Client(
        vertexai=True,
        project=service_info["project_id"],
        location=GEMINI_LOCATION,
        credentials=credentials,
    )
    _thread_local.client = client
    return client

def rewrite_paths():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input JSONL not found: {INPUT_JSONL}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total, changed = 0, 0
    with open(INPUT_JSONL, "r", encoding="utf-8") as src, open(PATH_UPDATED_JSONL, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip(): continue
            row = json.loads(line)
            total += 1
            old_path = str(row.get("chunk_path", ""))
            if OLD_AUDIO_PREFIX and old_path.startswith(OLD_AUDIO_PREFIX):
                row["chunk_path"] = NEW_AUDIO_PREFIX + old_path[len(OLD_AUDIO_PREFIX):]
                changed += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[step1] Completed. Rewrote {changed}/{total} paths.")

def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def parse_structured_response(response_text: str, expected_count: int) -> list[str]:
    data = BatchClassificationResponse.model_validate_json(response_text)
    if len(data.results) != expected_count:
        raise RuntimeError(f"Expected {expected_count} items, got {len(data.results)}")
    sorted_results = sorted(data.results, key=lambda x: x.item_index)
    return [res.intent.value for res in sorted_results]

def classify_batch(rows):
    lines = [f"Classify these {len(rows)} transcripts into their exact intent:"]
    for idx, row in enumerate(rows):
        lines.append(f"ITEM {idx}: transcript: {row.get('transcript', '')}")
    prompt = "\n".join(lines)
    client = get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=BatchClassificationResponse,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    return parse_structured_response(response.text, len(rows))

def classify_with_retries(rows):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return classify_batch(rows)
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt + random.uniform(0, 0.5))
    raise RuntimeError("Max retries failed")

completed_batches = 0
progress_lock = threading.Lock()

def process_batch(b_num, rows, total_batches):
    global completed_batches
    try:
        labels = classify_with_retries(rows)
    except Exception:
        labels = []
        for r in rows:
            old_intent = str(r.get("intent", ""))
            labels.append(old_intent if old_intent in INTENT_SET else IntentEnum.UNCLEAR_INPUT.value)
    
    with progress_lock:
        completed_batches += 1
        print(f"[step2] Completed Batch {completed_batches}/{total_batches}", flush=True)

    return rows, labels

def load_checkpoint():
    completed = set()
    if not CHECKPOINT_JSONL.exists():
        return completed
    with open(CHECKPOINT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    completed.add(int(json.loads(line)["_source_index"]))
                except Exception:
                    continue
    return completed

def relabel_and_rename():
    if not PATH_UPDATED_JSONL.exists():
        raise FileNotFoundError(f"Step 1 output missing: {PATH_UPDATED_JSONL}")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(PATH_UPDATED_JSONL)
    for index, row in enumerate(rows):
        row["_source_index"] = index

    checkpoint = load_checkpoint()
    ready = [r for r in rows if r["_source_index"] not in checkpoint and Path(str(r.get("chunk_path", ""))).exists()]

    if not ready:
        print("[step2] No items left to process.")
        return

    batches = [ready[i:i + BATCH_SIZE] for i in range(0, len(ready), BATCH_SIZE)]
    print(f"[step2] Processing {len(ready)} items across {len(batches)} batches...", flush=True)

    # Calculate starting ID once
    next_id = 1
    if FINAL_JSONL.exists():
        for r in load_jsonl(FINAL_JSONL):
            stem = Path(str(r.get("chunk_path", ""))).stem
            if "_" in stem and stem.rsplit("_", 1)[-1].isdigit():
                next_id = max(next_id, int(stem.rsplit("_", 1)[-1]) + 1)

    with open(FINAL_JSONL, "a", encoding="utf-8") as out_f, open(CHECKPOINT_JSONL, "a", encoding="utf-8") as chk_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
            futures = [executor.submit(process_batch, b_num, batch, len(batches)) for b_num, batch in enumerate(batches, 1)]
            
            for future in concurrent.futures.as_completed(futures):
                batch_rows, labels = future.result()
                for row, new_intent in zip(batch_rows, labels):
                    curr_audio = Path(str(row.get("chunk_path", "")))
                    if not curr_audio.exists():
                        continue
                    
                    suffix = curr_audio.suffix or ".wav"
                    new_audio_path = AUDIO_DIR / f"{new_intent}_{next_id:08d}{suffix}"
                    while new_audio_path.exists():
                        next_id += 1
                        new_audio_path = AUDIO_DIR / f"{new_intent}_{next_id:08d}{suffix}"

                    try:
                        curr_audio.rename(new_audio_path)
                    except Exception:
                        continue

                    row["intent"] = new_intent
                    row["chunk_path"] = str(new_audio_path)
                    idx = row.pop("_source_index")

                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    chk_f.write(json.dumps({"_source_index": idx}) + "\n")
                    next_id += 1

                out_f.flush()
                chk_f.flush()

    print("[step2] Relabeling and audio rename complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite-paths", action="store_true")
    parser.add_argument("--relabel", action="store_true")
    args = parser.parse_args()

    if args.rewrite_paths:
        rewrite_paths()
    if args.relabel:
        relabel_and_rename()
