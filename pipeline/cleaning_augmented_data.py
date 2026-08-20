"""Clean up the synthetic rows in call_trascript_intent_data_augmented/data.jsonl.

Two issues showed up in a manual review of generated sentences:

1. English words were deliberately kept in Latin script for TTS pronunciation
   (e.g. "जी, मैं note कर रियो हूँ।") - but the TRANSCRIPT label should be pure
   Devanagari, the way a Hindi ASR system/human transcriber would actually
   write it (e.g. "जी, मैं नोट कर रियो हूँ।"). The AUDIO is untouched - only
   the `transcript` text field is rewritten.
2. Some generated sentences have a stray nonsensical number in them (usually
   a leftover list-numbering artifact from generation, e.g.
   "74. ऐप में पेमेंट पेंडिंग दिख रहो हतो, फेल नाय; फिर फीस काहे") - these get
   removed. Numbers that make real sense in context (amounts, EMI counts,
   dates) are left alone.

Also fixes a naming bug in already-written rows: `conversation_id` had the
intent name duplicated (e.g. "synthetic_OBJECTION_ON_CHARGE_OBJECTION_ON_CHARGE_<uuid>"
instead of "synthetic_OBJECTION_ON_CHARGE_<uuid>") - collapsed here.

Sent to the LLM in batches of 100. Resumable: writes to data.cleaned.jsonl
alongside the original, resuming from however many lines are already there;
once every row is cleaned, atomically swaps it in as data.jsonl (original
backed up as data.jsonl.bak).

    .venv/bin/python -m pipeline.cleaning_augmented_data run
    .venv/bin/python -m pipeline.cleaning_augmented_data run --use-gemini
    .venv/bin/python -m pipeline.cleaning_augmented_data run --limit 500   # dry-run a sample first
"""

import argparse
import asyncio
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone

from pydantic import BaseModel

from . import config
from .intents import ClassificationError, IntentClassifier, OpenAIIntentClassifier

CLEAN_MODEL = config.AUGMENT_MODEL
CLEAN_BATCH_SIZE = 100
CLEAN_CONCURRENCY = 20


def _now():
    return datetime.now(timezone.utc).isoformat()


class CleanedItem(BaseModel):
    index: int
    cleaned_transcript: str


class CleanedBatch(BaseModel):
    items: list[CleanedItem]


CLEAN_SYSTEM_PROMPT = (
    "You are cleaning synthetic Hindi sentences generated for an EMI/loan-"
    "collections call dataset. Each sentence is what a borrower might say on "
    "such a call. Fix exactly two things, nothing else - do not rephrase, "
    "reorder, translate, or otherwise change the sentence:\n\n"
    "1. Some sentences have English words deliberately written in LATIN "
    "SCRIPT embedded in the Devanagari text (e.g. 'time', 'pay', 'note', "
    "'EMI', 'OK'). Convert EVERY such Latin-script word into its natural "
    "Devanagari phonetic transliteration, the way a Hindi ASR system or "
    "human transcriber would actually write it - e.g. 'note' -> 'नोट', "
    "'time' -> 'टाइम', 'pay' -> 'पे', 'EMI' -> 'ईएमआई', 'OK' -> 'ओके', "
    "'call' -> 'कॉल'. After cleaning, the sentence must be 100% Devanagari - "
    "no Latin letters anywhere.\n"
    "2. Some sentences have a stray, nonsensical number in them - usually a "
    "leftover list-numbering artifact (e.g. a sentence starting with '74. ' "
    "or '12) ' that has nothing to do with the sentence's meaning). Remove "
    "ONLY numbers like that. KEEP numbers that make real sense in context "
    "(rupee amounts, EMI counts, dates, durations, etc).\n\n"
    "If a sentence already has neither issue, return it completely unchanged. "
    "Return exactly one item per index, tagged with that same `index`."
)


def _clean_batch_prompt(transcripts):
    numbered = "\n".join(f'{i}: """{t}"""' for i, t in enumerate(transcripts))
    return (
        f"Clean each of the following {len(transcripts)} sentences independently "
        f"per the rules above. Return exactly {len(transcripts)} items.\n\n{numbered}"
    )


_DUP_INTENT_RE = re.compile(r"^synthetic_([A-Z_]+)_\1_")


def fix_conversation_id(rec):
    cid = rec.get("conversation_id") or ""
    m = _DUP_INTENT_RE.match(cid)
    if m:
        rec["conversation_id"] = "synthetic_" + m.group(1) + "_" + cid[m.end():]
    return rec


def _clean_batch_with_retries(classifier, transcripts):
    """Blocking call with retries, meant for asyncio.to_thread. Returns
    (cleaned_texts_or_None, error_info_or_None)."""
    last_error = None
    last_traceback = None
    total_attempts = 1 + config.CLASSIFY_MAX_RETRIES
    for attempt in range(1, total_attempts + 1):
        try:
            parsed = classifier.generate_structured(
                CLEAN_SYSTEM_PROMPT, _clean_batch_prompt(transcripts), CleanedBatch
            )
            by_index = {item.index: item.cleaned_transcript for item in parsed.items}
            missing = [i for i in range(len(transcripts)) if i not in by_index]
            if missing:
                raise ClassificationError(f"batch response missing indices: {missing}")
            return [by_index[i] for i in range(len(transcripts))], None
        except Exception as e:
            last_error = e
            last_traceback = traceback.format_exc()
            if attempt < total_attempts:
                print(
                    f"[cleaning] batch attempt {attempt}/{total_attempts} failed "
                    f"({type(e).__name__}: {e}), retrying in {config.CLASSIFY_RETRY_DELAY_S}s..."
                )
                time.sleep(config.CLASSIFY_RETRY_DELAY_S)

    error_info = {
        "batch_size": len(transcripts),
        "error": f"{type(last_error).__name__}: {last_error}",
        "attempts": total_attempts,
        "traceback": last_traceback,
        "time": _now(),
    }
    print(f"[cleaning] batch failed after {total_attempts} attempts: {error_info['error']}")
    return None, error_info


async def run(
    output_dir=config.OUTPUT_DIR_AUGMENTED,
    model=CLEAN_MODEL,
    batch_size=CLEAN_BATCH_SIZE,
    concurrency=CLEAN_CONCURRENCY,
    use_openai=True,
    limit=None,
):
    data_path = os.path.join(output_dir, config.DATA_FILENAME)
    cleaned_path = os.path.join(output_dir, "data.cleaned.jsonl")
    errors_path = os.path.join(output_dir, "cleaning_errors.log")

    with open(data_path, encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]

    already = 0
    if os.path.exists(cleaned_path):
        with open(cleaned_path, encoding="utf-8") as f:
            already = sum(1 for line in f if line.strip())
    print(f"[cleaning] {len(all_rows)} total rows, {already} already cleaned, resuming after those.")

    rows_to_process = all_rows[already:]
    if limit is not None:
        rows_to_process = rows_to_process[:limit]

    classifier = OpenAIIntentClassifier(model=model) if use_openai else IntentClassifier()
    print(f"[cleaning] backend: {classifier.name} ({classifier.model})")

    processed_this_run = 0
    stopped = False

    with open(cleaned_path, "a", encoding="utf-8") as out_f, open(errors_path, "a", encoding="utf-8") as err_f:
        for group_start in range(0, len(rows_to_process), batch_size * concurrency):
            group = rows_to_process[group_start : group_start + batch_size * concurrency]
            sub_batches = [group[i : i + batch_size] for i in range(0, len(group), batch_size)]

            tasks = [
                asyncio.to_thread(_clean_batch_with_retries, classifier, [r["transcript"] for r in sb])
                for sb in sub_batches
            ]
            results = await asyncio.gather(*tasks)

            any_failed = False
            for sb, (cleaned_texts, error_info) in zip(sub_batches, results):
                if cleaned_texts is None:
                    err_f.write(json.dumps(error_info) + "\n")
                    err_f.flush()
                    any_failed = True
                    continue
                for rec, cleaned_text in zip(sb, cleaned_texts):
                    final_record = fix_conversation_id(dict(rec))
                    final_record["transcript"] = cleaned_text.strip()
                    out_f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
                processed_this_run += len(sb)
            out_f.flush()

            print(f"[cleaning] {already + processed_this_run}/{len(all_rows)} rows cleaned so far")

            if any_failed:
                stopped = True
                print("[cleaning] STOPPED: a batch failed - re-run the same command to resume.")
                break

    total_done = already + processed_this_run
    if not stopped and total_done >= len(all_rows) and limit is None:
        backup_path = data_path + ".bak"
        os.replace(data_path, backup_path)
        os.replace(cleaned_path, data_path)
        print(f"[cleaning] all {total_done} rows cleaned - swapped into {data_path} (original backed up at {backup_path})")
    else:
        print(f"[cleaning] {total_done}/{len(all_rows)} done, not yet swapped in - re-run to continue")

    return {"total_rows": len(all_rows), "processed_this_run": processed_this_run, "total_done": total_done, "stopped": stopped}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Clean transcripts + fix conversation_id in the augmented dataset")
    run_p.add_argument("--output-dir", default=config.OUTPUT_DIR_AUGMENTED)
    run_p.add_argument("--model", default=CLEAN_MODEL)
    run_p.add_argument("--batch-size", type=int, default=CLEAN_BATCH_SIZE)
    run_p.add_argument("--concurrency", type=int, default=CLEAN_CONCURRENCY)
    run_p.add_argument("--use-gemini", action="store_true", help="use Gemini instead of the OpenAI default")
    run_p.add_argument("--limit", type=int, default=None, help="only clean the first N not-yet-cleaned rows (dry run)")
    args = parser.parse_args()

    asyncio.run(
        run(
            output_dir=args.output_dir,
            model=args.model,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            use_openai=not args.use_gemini,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
