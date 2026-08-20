"""Build v3 of the dataset: re-classify every existing transcript under the
new, consolidated 20-intent taxonomy (see intents.md) using OpenAI.

No transcription happens here - v1 and v2 already collected every transcript.
This only recomputes `intent` for each row, in batches of 100, and writes the
result to call_trascript_intent_data_v3/data.jsonl. Input is v1
(call_trascript_intent_data) + v2 (call_trascript_intent_data_v2)'s data.jsonl,
deduped by chunk_path - the same corpus already pushed to the Hub, read
locally rather than re-downloading audio that isn't changing.

Batches run concurrently (default: 10 at a time, 100 rows each = 1000 rows in
flight) since the only bottleneck here is LLM round-trip latency, not local
CPU/disk work - each batch is a blocking OpenAI call run in its own thread via
asyncio.to_thread, awaited together with asyncio.gather.

Resumable: re-running skips chunk_paths already present in v3's data.jsonl.
If any batch in a concurrent group exhausts its config.CLASSIFY_MAX_RETRIES
retries, the run stops after that group - batches that succeeded are already
written; the failed one(s) simply weren't, so they're picked up again next run.

    .venv/bin/python -m pipeline.build_v3 run
    .venv/bin/python -m pipeline.build_v3 run --limit 5000
    .venv/bin/python -m pipeline.build_v3 run --concurrency 20

Then push v3 to the Hub (replaces the "train" split's intent labels - this is
a full relabel of the same rows, not an append):

    .venv/bin/python -m pipeline.push_to_hub --repo-id <repo> --output-dir data/call_trascript_intent_data_v3
"""

import argparse
import asyncio
import json
import os
import time
import traceback
from datetime import datetime, timezone

from . import config
from .intents import BatchIntentResult, ClassificationError, OpenAIIntentClassifier, build_system_prompt

V3_MODEL = "gpt-5.6-luna"
V3_BATCH_SIZE = 100
V3_CONCURRENCY = 20


def _now():
    return datetime.now(timezone.utc).isoformat()


def _iter_source_rows(paths):
    """Yield rows deduped by chunk_path across multiple v1/v2 data.jsonl files,
    with the old `intent` field dropped - it gets freshly recomputed."""
    seen = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                chunk_path = rec.get("chunk_path")
                if not chunk_path or chunk_path in seen:
                    continue
                seen.add(chunk_path)
                rec.pop("intent", None)
                yield rec


def _read_done_chunk_paths(data_path):
    done = set()
    if os.path.exists(data_path):
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["chunk_path"])
                except Exception:
                    continue
    return done


def _v3_batch_prompt(transcripts):
    numbered = "\n".join(f'{i}: """{t}"""' for i, t in enumerate(transcripts))
    return (
        f"You are re-classifying {len(transcripts)} transcripts under a freshly "
        "consolidated, closed 20-intent taxonomy (defined above) for this EMI/"
        "loan-collections call context. Every transcript gets EXACTLY one of "
        "those 20 intents - this taxonomy must not be grown, only these 20 exist.\n\n"
        "Prefer a specific intent whenever there is ANY discernible signal - only "
        "fall back to UNCLEAR_INPUT when the transcript is genuinely "
        "unintelligible, off-topic/crosstalk, or empty/pure noise. Do not default "
        "to UNCLEAR_INPUT just because an utterance is short: a bare "
        "acknowledgement, greeting, or one-word answer usually DOES fit a "
        "specific intent (e.g. GENERAL_AFFIRMATIVE_ACKNOWLEDGEMENT, "
        "CALL_OPENING_OR_PROMPT). Try to classify in other than (GENERAL_AFFIRMATIVE_ACKNOWLEDGEMENT and UNCLEAR_INPUT) if possible if not check GENERAL_AFFIRMATIVE_ACKNOWLEDGEMENT if it doesn't fall in this then UNCLEAR_INPUT\n\n"
        f"Return exactly one item per index, {len(transcripts)} items total.\n\n"
        f"{numbered}"
    )


def _classify_batch_with_retries(classifier, system_prompt, transcripts):
    """Blocking call with retries - meant to be run via asyncio.to_thread so
    concurrent batches overlap. Returns (intents_or_None, error_info_or_None).
    Does its own file I/O for NOTHING - the caller logs/writes, so this never
    touches a shared file handle from a worker thread."""
    last_error = None
    last_traceback = None
    total_attempts = 1 + config.CLASSIFY_MAX_RETRIES
    for attempt in range(1, total_attempts + 1):
        try:
            parsed = classifier.generate_structured(
                system_prompt, _v3_batch_prompt(transcripts), BatchIntentResult
            )
            by_index = {item.index: item.intent.value for item in parsed.items}
            missing = [i for i in range(len(transcripts)) if i not in by_index]
            if missing:
                raise ClassificationError(f"batch response missing indices: {missing}")
            return [by_index[i] for i in range(len(transcripts))], None
        except Exception as e:
            last_error = e
            last_traceback = traceback.format_exc()
            if attempt < total_attempts:
                print(
                    f"[build_v3] batch attempt {attempt}/{total_attempts} failed "
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
    print(f"[build_v3] batch failed after {total_attempts} attempts: {error_info['error']}")
    return None, error_info


def write_metadata(output_dir, **stats):
    from .intents import INTENT_NAMES

    meta = {
        "dataset_name": os.path.basename(output_dir),
        "description": (
            "v3: full re-classification of the call_trascript_intent_data(_v2) "
            "corpus under a consolidated, closed 20-intent taxonomy for EMI/loan "
            "collections calls. Same audio/transcripts as v1+v2 - only `intent` "
            "was recomputed."
        ),
        "source_dirs": [config.OUTPUT_DIR, config.OUTPUT_DIR_V2],
        "intent_taxonomy_source": config.INTENTS_MD_PATH,
        "intents": INTENT_NAMES,
        "intent_model": f"openai:{V3_MODEL}",
        "intent_batch_size": V3_BATCH_SIZE,
        "last_updated": _now(),
    }
    meta.update(stats)
    tmp_path = os.path.join(output_dir, config.METADATA_FILENAME + ".tmp")
    final_path = os.path.join(output_dir, config.METADATA_FILENAME)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, final_path)


async def run(
    limit=None,
    output_dir=config.OUTPUT_DIR_V3,
    model=V3_MODEL,
    batch_size=V3_BATCH_SIZE,
    concurrency=V3_CONCURRENCY,
):
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, config.DATA_FILENAME)
    errors_path = os.path.join(output_dir, config.ERRORS_LOG_FILENAME)

    done = _read_done_chunk_paths(data_path)
    print(f"[build_v3] {len(done)} chunks already reclassified, resuming after those.")

    source_paths = [
        os.path.join(config.OUTPUT_DIR, config.DATA_FILENAME),
        os.path.join(config.OUTPUT_DIR_V2, config.DATA_FILENAME),
    ]

    classifier = OpenAIIntentClassifier(model=model)
    system_prompt = build_system_prompt()
    print(
        f"[build_v3] intent backend: openai ({model}), batch_size={batch_size}, "
        f"concurrency={concurrency} ({batch_size * concurrency} rows/group)"
    )

    write_metadata(output_dir, status="in_progress", num_chunks_processed=len(done))

    processed_this_run = 0
    stopped = False
    stop_reason = None
    group = []  # up to batch_size * concurrency records awaiting classification

    async def flush_group(data_f, err_f):
        nonlocal processed_this_run, stopped, stop_reason
        if not group:
            return True

        sub_batches = [group[i : i + batch_size] for i in range(0, len(group), batch_size)]
        tasks = [
            asyncio.to_thread(
                _classify_batch_with_retries, classifier, system_prompt, [r["transcript"] for r in sb]
            )
            for sb in sub_batches
        ]
        results = await asyncio.gather(*tasks)

        any_failed = False
        for sb, (intents_out, error_info) in zip(sub_batches, results):
            if intents_out is None:
                err_f.write(json.dumps(error_info) + "\n")
                err_f.flush()
                any_failed = True
                if stop_reason is None:
                    stop_reason = error_info["error"]
                continue
            for rec, intent in zip(sb, intents_out):
                final_record = dict(rec)
                final_record["intent"] = intent
                data_f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
            processed_this_run += len(sb)
        data_f.flush()

        print(
            f"[build_v3] group: {len(sub_batches)} batches, "
            f"{sum(1 for r in results if r[0] is not None)} ok, "
            f"{sum(1 for r in results if r[0] is None)} failed "
            f"({len(done) + processed_this_run} total done)"
        )
        write_metadata(
            output_dir, status="in_progress", num_chunks_processed=len(done) + processed_this_run
        )
        group.clear()

        if any_failed:
            stopped = True
            print("[build_v3] STOPPED: at least one batch in the last group failed. Re-run to resume.")
            return False
        return True

    with open(data_path, "a", encoding="utf-8") as data_f, open(errors_path, "a", encoding="utf-8") as err_f:
        for rec in _iter_source_rows(source_paths):
            if rec["chunk_path"] in done:
                continue
            if limit is not None and processed_this_run + len(group) >= limit:
                break

            if not (rec.get("transcript") or "").strip():
                # No signal to classify - deterministic, no need to spend a call.
                final_record = dict(rec)
                final_record["intent"] = "UNCLEAR_INPUT"
                data_f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
                data_f.flush()
                processed_this_run += 1
                continue

            group.append(rec)
            if len(group) >= batch_size * concurrency:
                if not await flush_group(data_f, err_f):
                    break

        if not stopped:
            await flush_group(data_f, err_f)

    write_metadata(
        output_dir,
        status="stopped_on_error" if stopped else "complete" if limit is None else "paused",
        num_chunks_processed=len(done) + processed_this_run,
        last_error=stop_reason,
    )

    result = {
        "processed_this_run": processed_this_run,
        "total_done": len(done) + processed_this_run,
        "stopped": stopped,
        "stop_reason": stop_reason,
    }
    print(f"[build_v3] done -> {result}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Run/resume the v3 re-classification")
    run_p.add_argument("--limit", type=int, default=None, help="Cap on number of NEW chunks this call")
    run_p.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR_V3)
    run_p.add_argument("--model", type=str, default=V3_MODEL)
    run_p.add_argument("--batch-size", type=int, default=V3_BATCH_SIZE)
    run_p.add_argument(
        "--concurrency", type=int, default=V3_CONCURRENCY, help="Number of batches classified concurrently"
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            limit=args.limit,
            output_dir=args.output_dir,
            model=args.model,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
