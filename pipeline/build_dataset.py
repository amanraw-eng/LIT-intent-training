"""Build the call_trascript_intent_data dataset: transcript + intent per audio chunk.

Run with the venv's python as a module (relative imports require -m):

    .venv/bin/python -m pipeline.build_dataset sample            # first 10 chunks, for verification
    .venv/bin/python -m pipeline.build_dataset sample --n 25      # first 25 chunks
    .venv/bin/python -m pipeline.build_dataset run                # full run, resumable
    .venv/bin/python -m pipeline.build_dataset run --limit 500     # process at most 500 new chunks this call
    .venv/bin/python -m pipeline.build_dataset retry-skipped       # retry chunks that were skipped

Add --use-openai to any command to classify intent with OpenAI instead of the
default Gemini backend (reads OPENAI_API_KEY from .env in the project root):

    .venv/bin/python -m pipeline.build_dataset run --use-openai

Resuming: every successfully processed chunk is appended to data.jsonl immediately.
Re-running `run` reads that file, skips chunk_paths already present, and continues.

Intent classification is batched: transcribed chunks are buffered (durably, in
pending_intent.jsonl) until config.INTENT_BATCH_SIZE are ready, then classified
with a single LLM call instead of one call per chunk. A stop mid-batch keeps the
already-transcribed chunks in pending_intent.jsonl so no transcription work is
lost - the next run classifies them together with newly transcribed ones.

Two different kinds of trouble are handled differently:
  - Any transcription-side failure (the server hangs after flush, a connection
    drops, DNS blips, an explicit server error) is chunk-specific noise: it's
    logged to skipped.jsonl and the run continues with the next chunk. If
    MAX_CONSECUTIVE_TRANSCRIPTION_FAILURES happen in a row, that's treated as a
    real outage instead and the run stops (see config.py).
  - Gemini/OpenAI intent-classification failures (quota, auth, unexpected
    exceptions) STOP the run immediately - continuing would just burn through
    the rest of the dataset marking everything as failed. Progress already
    written (and the pending batch) is safe; just run `run` again to resume.
"""

import argparse
import asyncio
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone

from datasets import load_from_disk

from . import config, transcribe
from .intents import INTENT_NAMES, build_classifier
from .transcribe import TranscriptionError

CHUNK_FILENAME_RE = re.compile(r"(\d+)\.wav$")


def iter_flat_chunks(ds):
    """Flatten conversation-level rows into one dict per audio chunk, chronological per call.

    The source dataset's chunk_paths list is NOT positionally aligned with its
    timestamps list - each chunk file's name is the index into `timestamps`
    (verified against actual per-file audio duration), and start/end there are
    sample counts at config.SOURCE_SAMPLE_RATE, not milliseconds.
    """
    for row in ds:
        chunk_paths = json.loads(row["chunk_paths"])
        timestamps = json.loads(row["timestamps"])

        resolved = []
        for path in chunk_paths:
            m = CHUNK_FILENAME_RE.search(os.path.basename(path))
            if not m:
                continue
            ts = timestamps[int(m.group(1))]
            resolved.append((path, ts))
        resolved.sort(key=lambda p: p[1]["start"])

        for idx, (path, ts) in enumerate(resolved):
            samples_per_ms = config.SOURCE_SAMPLE_RATE / 1000.0
            yield {
                "oid": row["oid"],
                "conversation_id": row["conversation_id"],
                "recording_url": row["recording_url"],
                "chunk_path": path,
                "chunk_index": idx,
                "start_ms": round(ts["start"] / samples_per_ms, 1),
                "end_ms": round(ts["end"] / samples_per_ms, 1),
            }


def compute_totals(ds):
    num_chunks = 0
    for row in ds:
        num_chunks += len(json.loads(row["chunk_paths"]))
    return len(ds), num_chunks


def _read_chunk_paths(path):
    paths = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    paths.add(json.loads(line)["chunk_path"])
                except Exception:
                    continue
    return paths


def _count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_processed_paths(data_path, skipped_path, pending_path):
    """Chunks already done, skipped, or awaiting classification - all excluded
    from re-transcription on a normal `run`."""
    return (
        _read_chunk_paths(data_path)
        | _read_chunk_paths(skipped_path)
        | _read_chunk_paths(pending_path)
    )


def load_excluded_paths(other_output_dirs):
    """Chunks already done or pending classification in OTHER output dirs (e.g.
    v1 while running v2) - excluded so this run continues past them instead of
    redoing the work. Deliberately does NOT include those dirs' skipped.jsonl:
    a chunk that failed there is fair game to retry here, e.g. with a different
    transcription backend that might actually handle it."""
    excluded = set()
    for d in other_output_dirs:
        excluded |= _read_chunk_paths(os.path.join(d, config.DATA_FILENAME))
        excluded |= _read_chunk_paths(os.path.join(d, config.PENDING_INTENT_FILENAME))
    return excluded


def _load_pending(pending_path):
    pending = []
    if os.path.exists(pending_path):
        with open(pending_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pending.append(json.loads(line))
                    except Exception:
                        continue
    return pending


def _now():
    return datetime.now(timezone.utc).isoformat()


def _rewrite_pending_file(pending_path, pending_buffer):
    tmp_path = pending_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in pending_buffer:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, pending_path)


def _classify_one_batch(classifier, transcripts, err_f, pending_path):
    """Single classify_batch call with retries. Returns (intents_or_None, stop_reason)."""
    last_error = None
    last_traceback = None
    total_attempts = 1 + config.CLASSIFY_MAX_RETRIES
    for attempt in range(1, total_attempts + 1):
        try:
            return classifier.classify_batch(transcripts), None
        except Exception as e:
            last_error = e
            last_traceback = traceback.format_exc()
            if attempt < total_attempts:
                print(
                    f"[build_dataset] batch classification attempt {attempt}/{total_attempts} "
                    f"failed ({type(e).__name__}: {e}), retrying in {config.CLASSIFY_RETRY_DELAY_S}s..."
                )
                time.sleep(config.CLASSIFY_RETRY_DELAY_S)

    stop_reason = f"{type(last_error).__name__}: {last_error}"
    err_f.write(
        json.dumps(
            {
                "batch_size": len(transcripts),
                "error": stop_reason,
                "attempts": total_attempts,
                "traceback": last_traceback,
                "time": _now(),
            }
        )
        + "\n"
    )
    err_f.flush()
    print(
        f"[build_dataset] STOPPED on batch classification error after {total_attempts} "
        f"attempts: {stop_reason}"
    )
    print(
        f"[build_dataset] Progress saved (transcripts kept in "
        f"{os.path.basename(pending_path)}). Re-run to resume from here."
    )
    return None, stop_reason


def _flush_pending(classifier, pending_buffer, data_f, err_f, pending_path):
    """Classify buffered transcripts in sub-batches of at most
    config.INTENT_BATCH_SIZE each (pending_buffer can be larger than that - e.g.
    if it accumulated across several earlier runs before ever hitting the
    threshold, or the batch size was lowered after some items were already
    queued - sending it all in one oversized call is what makes Gemini's
    structured output truncate and fail to parse).

    Sub-batches are classified in order. On the first failure, everything
    classified so far this call is already written to data.jsonl and dropped
    from pending_buffer/pending_path; the failed sub-batch and everything after
    it stays in pending_buffer/pending_path for the next run. Returns
    (ok, stop_reason, n_flushed).
    """
    total_flushed = 0
    while pending_buffer:
        batch = pending_buffer[: config.INTENT_BATCH_SIZE]
        n = len(batch)
        transcripts = [rec["transcript"] for rec in batch]

        batch_intents, stop_reason = _classify_one_batch(classifier, transcripts, err_f, pending_path)
        if batch_intents is None:
            _rewrite_pending_file(pending_path, pending_buffer)
            return False, stop_reason, total_flushed

        for rec, intent in zip(batch, batch_intents):
            final_record = dict(rec)
            final_record["intent"] = intent
            data_f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
        data_f.flush()
        print(f"[build_dataset] classified batch of {n}")
        total_flushed += n

        del pending_buffer[:n]
        _rewrite_pending_file(pending_path, pending_buffer)

    return True, None, total_flushed


def write_metadata(output_dir, **stats):
    meta = {
        "dataset_name": os.path.basename(output_dir),
        "description": (
            "Per-chunk ASR transcripts and closed-set intent labels for sampled "
            "loan-collection calls, derived from call_sampled_50_hours_with_chunks."
        ),
        "source_dataset_dir": config.SOURCE_DATASET_DIR,
        "features": {
            "oid": "string - source conversation object id",
            "conversation_id": "string - source conversation id",
            "recording_url": "string - original full call recording url",
            "chunk_path": "string - local path to the chunk wav file (8kHz mono PCM16)",
            "chunk_index": "int - chunk order within the conversation, sorted by start_ms",
            "start_ms": "float - chunk start offset in the source recording (ms)",
            "end_ms": "float - chunk end offset in the source recording (ms)",
            "duration_s": "float - chunk duration in seconds",
            "transcript": "string - ASR transcript from the LIT websocket service",
            "intent": "string - one of the closed-set intents in intents.md",
        },
        "intents": INTENT_NAMES,
        "transcription_endpoint": config.LIT_WS_BASE,
        "transcription_language": config.LANGUAGE,
        "intent_taxonomy_source": config.INTENTS_MD_PATH,
        "intent_batch_size": config.INTENT_BATCH_SIZE,
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
    output_dir=config.OUTPUT_DIR,
    reset=False,
    use_openai=False,
    use_vllm=False,
    exclude_dirs=(),
):
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, config.DATA_FILENAME)
    errors_path = os.path.join(output_dir, config.ERRORS_LOG_FILENAME)
    skipped_path = os.path.join(output_dir, config.SKIPPED_FILENAME)
    pending_path = os.path.join(output_dir, config.PENDING_INTENT_FILENAME)

    if reset:
        for p in (data_path, skipped_path, pending_path):
            if os.path.exists(p):
                os.remove(p)

    seen = load_processed_paths(data_path, skipped_path, pending_path)
    num_skipped_prior = len(_read_chunk_paths(skipped_path))
    num_processed_prior = _count_lines(data_path)
    pending_buffer = _load_pending(pending_path)
    if pending_buffer:
        print(f"[build_dataset] {len(pending_buffer)} chunks already transcribed and awaiting classification")
    print(f"[build_dataset] {len(seen)} chunks already done/skipped/pending, resuming after those.")

    if exclude_dirs:
        excluded = load_excluded_paths(exclude_dirs)
        seen |= excluded
        print(
            f"[build_dataset] excluding {len(excluded)} chunks already done/pending in "
            f"{list(exclude_dirs)} (their skipped chunks are still retried here)"
        )

    ds = load_from_disk(config.SOURCE_DATASET_DIR)
    num_conversations, num_chunks_total = compute_totals(ds)
    print(f"[build_dataset] source: {num_conversations} conversations, {num_chunks_total} chunks total")

    transcriber = transcribe.build_transcriber(use_vllm=use_vllm)
    print(f"[build_dataset] transcription backend: {transcriber.name}")
    await transcriber.prepare()

    classifier = build_classifier(use_openai=use_openai)
    print(f"[build_dataset] intent backend: {classifier.name} ({classifier.model})")

    def _meta_stats():
        return dict(
            status="in_progress",
            num_conversations_total=num_conversations,
            num_chunks_total=num_chunks_total,
            num_chunks_processed=num_processed_prior + processed_this_run,
            num_chunks_skipped=num_skipped_prior + skipped_this_run,
            num_chunks_pending=len(pending_buffer),
            transcription_backend=transcriber.name,
            intent_backend=classifier.name,
            intent_model=f"{classifier.name}:{classifier.model}",
        )

    processed_this_run = 0
    skipped_this_run = 0
    # Incremented the moment a chunk is transcribed (whether finalized right
    # away or queued in pending_buffer) or skip-logged - i.e. "no longer left
    # to do", regardless of whether its intent has been classified yet. This is
    # what --limit counts against; processed_this_run alone lags behind while
    # a batch is still buffered, which would let a run blow past --limit.
    handled_this_run = 0
    consecutive_transcription_failures = 0
    stopped = False
    stopped_on_classification = False
    stop_reason = None

    write_metadata(output_dir, **_meta_stats())

    with open(data_path, "a", encoding="utf-8") as data_f, open(
        errors_path, "a", encoding="utf-8"
    ) as err_f, open(skipped_path, "a", encoding="utf-8") as skip_f:
        for item in iter_flat_chunks(ds):
            if item["chunk_path"] in seen:
                continue
            if limit is not None and handled_this_run >= limit:
                break

            call_id = f"gen-{item['conversation_id']}-{item['chunk_index']}"

            # Transcription failures (hangs, connection drops, DNS blips, server
            # errors) are chunk-specific noise - skip and keep going. A long run
            # of consecutive failures instead suggests a real outage, so bail out
            # rather than silently skipping the rest of the dataset unattended.
            try:
                transcript, duration_s = await transcriber.transcribe_chunk(item["chunk_path"], call_id)
            except TranscriptionError as e:
                consecutive_transcription_failures += 1
                skip_f.write(
                    json.dumps(
                        {
                            "chunk_path": item["chunk_path"],
                            "reason": str(e),
                            "error_type": type(e).__name__,
                            "time": _now(),
                        }
                    )
                    + "\n"
                )
                skip_f.flush()
                seen.add(item["chunk_path"])
                skipped_this_run += 1
                handled_this_run += 1
                print(f"[build_dataset] SKIPPED ({type(e).__name__}) {item['chunk_path']}: {e}")

                if consecutive_transcription_failures >= config.MAX_CONSECUTIVE_TRANSCRIPTION_FAILURES:
                    stop_reason = (
                        f"{consecutive_transcription_failures} consecutive transcription "
                        "failures - likely a service outage, stopping instead of skipping "
                        "the rest of the dataset"
                    )
                    print(f"[build_dataset] STOPPED: {stop_reason}")
                    print("[build_dataset] Progress saved. Re-run `run` to resume from here.")
                    stopped = True
                    break

                await asyncio.sleep(config.INTER_CHUNK_SLEEP_S)
                continue

            consecutive_transcription_failures = 0
            seen.add(item["chunk_path"])
            record_partial = {**item, "duration_s": duration_s, "transcript": transcript}

            if not transcript.strip():
                # No need to spend an LLM call on this - empty transcript is noise/silence.
                record_partial["intent"] = "SILENCE_NOISE"
                data_f.write(json.dumps(record_partial, ensure_ascii=False) + "\n")
                data_f.flush()
                processed_this_run += 1
            else:
                pending_buffer.append(record_partial)
                with open(pending_path, "a", encoding="utf-8") as pf:
                    pf.write(json.dumps(record_partial, ensure_ascii=False) + "\n")

                if len(pending_buffer) >= config.INTENT_BATCH_SIZE:
                    ok, reason, n = _flush_pending(classifier, pending_buffer, data_f, err_f, pending_path)
                    processed_this_run += n
                    if not ok:
                        stop_reason = reason
                        stopped = True
                        stopped_on_classification = True
                        break
            handled_this_run += 1

            if handled_this_run % 20 == 0:
                write_metadata(output_dir, **_meta_stats())
                print(
                    f"[build_dataset] {num_processed_prior + processed_this_run}/{num_chunks_total} "
                    f"done, {num_skipped_prior + skipped_this_run} skipped, {len(pending_buffer)} pending"
                )

            await asyncio.sleep(config.INTER_CHUNK_SLEEP_S)

        if pending_buffer and not stopped_on_classification:
            ok, reason, n = _flush_pending(classifier, pending_buffer, data_f, err_f, pending_path)
            processed_this_run += n
            if not ok:
                stop_reason = reason
                stopped = True
                stopped_on_classification = True

    final_status = "stopped_on_error" if stopped else ("complete" if limit is None else "paused")
    write_metadata(output_dir, status=final_status, last_error=stop_reason, **{
        k: v for k, v in _meta_stats().items() if k != "status"
    })

    return {
        "processed_this_run": processed_this_run,
        "skipped_this_run": skipped_this_run,
        "total_done": num_processed_prior + processed_this_run,
        "total_skipped": num_skipped_prior + skipped_this_run,
        "pending_unclassified": len(pending_buffer),
        "num_chunks_total": num_chunks_total,
        "stopped": stopped,
        "stop_reason": stop_reason,
    }


async def retry_skipped(output_dir=config.OUTPUT_DIR, limit=None, use_openai=False, use_vllm=False):
    """Re-attempt chunks in skipped.jsonl with a longer flush timeout.

    Successful transcriptions join the same batched-classification pending
    queue as `run` (pending_intent.jsonl), so a stop here is just as resumable.
    Chunks that still don't respond stay in skipped.jsonl.
    """
    data_path = os.path.join(output_dir, config.DATA_FILENAME)
    skipped_path = os.path.join(output_dir, config.SKIPPED_FILENAME)
    errors_path = os.path.join(output_dir, config.ERRORS_LOG_FILENAME)
    pending_path = os.path.join(output_dir, config.PENDING_INTENT_FILENAME)

    to_retry = []
    if os.path.exists(skipped_path):
        with open(skipped_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        to_retry.append(json.loads(line))
                    except Exception:
                        continue

    if limit is not None:
        to_retry = to_retry[:limit]

    pending_buffer = _load_pending(pending_path)
    if pending_buffer:
        print(f"[build_dataset] also classifying {len(pending_buffer)} chunks already pending from a previous run")

    print(f"[build_dataset] retrying {len(to_retry)} previously-skipped chunks")
    if not to_retry and not pending_buffer:
        return {"retried": 0, "recovered": 0, "still_skipped": 0, "pending_unclassified": 0}

    ds = load_from_disk(config.SOURCE_DATASET_DIR)
    by_path = {item["chunk_path"]: item for item in iter_flat_chunks(ds)}

    transcriber = transcribe.build_transcriber(use_vllm=use_vllm)
    print(f"[build_dataset] transcription backend: {transcriber.name}")
    await transcriber.prepare()

    classifier = build_classifier(use_openai=use_openai)
    print(f"[build_dataset] intent backend: {classifier.name} ({classifier.model})")

    still_skipped = []
    recovered = 0
    stopped = False
    stop_reason = None

    with open(data_path, "a", encoding="utf-8") as data_f, open(
        errors_path, "a", encoding="utf-8"
    ) as err_f:
        for entry in to_retry:
            chunk_path = entry["chunk_path"]
            item = by_path.get(chunk_path)
            if item is None:
                still_skipped.append(entry)
                continue

            call_id = f"retry-{item['conversation_id']}-{item['chunk_index']}-{int(time.time())}"
            try:
                transcript, duration_s = await transcriber.transcribe_chunk(
                    chunk_path,
                    call_id,
                    flush_timeout_s=config.RETRY_FLUSH_RECV_TIMEOUT_S,
                )
            except TranscriptionError as e:
                still_skipped.append(
                    {**entry, "reason": str(e), "error_type": type(e).__name__, "time": _now()}
                )
                print(f"[build_dataset] still failing ({type(e).__name__}): {chunk_path}")
                await asyncio.sleep(config.INTER_CHUNK_SLEEP_S)
                continue

            record_partial = {**item, "duration_s": duration_s, "transcript": transcript}
            if not transcript.strip():
                record_partial["intent"] = "SILENCE_NOISE"
                data_f.write(json.dumps(record_partial, ensure_ascii=False) + "\n")
                data_f.flush()
                recovered += 1
                print(f"[build_dataset] recovered: {chunk_path}")
            else:
                pending_buffer.append(record_partial)
                with open(pending_path, "a", encoding="utf-8") as pf:
                    pf.write(json.dumps(record_partial, ensure_ascii=False) + "\n")
                print(f"[build_dataset] recovered (pending classification): {chunk_path}")

                if len(pending_buffer) >= config.INTENT_BATCH_SIZE:
                    ok, reason, n = _flush_pending(classifier, pending_buffer, data_f, err_f, pending_path)
                    recovered += n
                    if not ok:
                        stop_reason = reason
                        stopped = True
                        break

            await asyncio.sleep(config.INTER_CHUNK_SLEEP_S)

        if pending_buffer and not stopped:
            ok, reason, n = _flush_pending(classifier, pending_buffer, data_f, err_f, pending_path)
            recovered += n
            if not ok:
                stop_reason = reason
                stopped = True

    # Rewrite skipped.jsonl: entries we attempted this call are resolved (either
    # recovered/pending, or re-added to still_skipped) - keep everything else as-is.
    attempted_paths = {e["chunk_path"] for e in to_retry}
    remaining = list(still_skipped)
    if os.path.exists(skipped_path):
        with open(skipped_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry["chunk_path"] not in attempted_paths:
                    remaining.append(entry)

    tmp_path = skipped_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for entry in remaining:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp_path, skipped_path)

    return {
        "retried": len(to_retry),
        "recovered": recovered,
        "still_skipped": len(remaining),
        "pending_unclassified": len(pending_buffer),
        "stopped": stopped,
        "stop_reason": stop_reason,
    }


def run_sample(n=10, output_dir=config.SAMPLE_OUTPUT_DIR, use_openai=False, use_vllm=False, exclude_dirs=()):
    """Verification helper: (re)generate just the first `n` chunks into a separate sample dir.

    Safe to call repeatedly - each call starts fresh (does not touch the main
    resumable dataset in call_trascript_intent_data).
    """
    return asyncio.run(
        run(
            limit=n,
            output_dir=output_dir,
            reset=True,
            use_openai=use_openai,
            use_vllm=use_vllm,
            exclude_dirs=exclude_dirs,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_backend_flags(p):
        p.add_argument(
            "--use-openai",
            action="store_true",
            help="Classify intent with OpenAI instead of Gemini (reads OPENAI_API_KEY from .env)",
        )
        p.add_argument(
            "--use-vllm",
            action="store_true",
            help=(
                "Transcribe with vLLM instead of the LIT websocket. Uses the ngrok "
                "tunnel at NGROK_ENDPOINT (.env) + /qwen-asr/v1/audio/transcriptions "
                "if set, else falls back to a local server at VLLM_BASE_URL "
                "(default http://localhost:5500) + /v1/audio/transcriptions"
            ),
        )

    def add_exclude_dir_flag(p):
        p.add_argument(
            "--exclude-dir",
            action="append",
            default=[],
            metavar="DIR",
            help=(
                "Skip chunks already done or pending classification in this OTHER "
                "output dir (repeatable) - e.g. point a v2 run at v1's dir so it "
                "continues past everything v1 already finished, without redoing it. "
                "Chunks v1 SKIPPED are still retried here."
            ),
        )

    sample_p = sub.add_parser("sample", help="Run a quick sample for verification")
    sample_p.add_argument("--n", type=int, default=10)
    add_backend_flags(sample_p)
    add_exclude_dir_flag(sample_p)

    run_p = sub.add_parser("run", help="Run/resume the full generation")
    run_p.add_argument(
        "--limit", type=int, default=None, help="Cap on number of NEW chunks to process this call"
    )
    run_p.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR)
    add_backend_flags(run_p)
    add_exclude_dir_flag(run_p)

    retry_p = sub.add_parser("retry-skipped", help="Retry chunks that were previously skipped")
    retry_p.add_argument("--limit", type=int, default=None)
    retry_p.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR)
    add_backend_flags(retry_p)

    args = parser.parse_args()

    start = time.time()
    if args.command == "sample":
        result = run_sample(
            n=args.n,
            use_openai=args.use_openai,
            use_vllm=args.use_vllm,
            exclude_dirs=args.exclude_dir,
        )
    elif args.command == "retry-skipped":
        result = asyncio.run(
            retry_skipped(
                output_dir=args.output_dir,
                limit=args.limit,
                use_openai=args.use_openai,
                use_vllm=args.use_vllm,
            )
        )
    else:
        result = asyncio.run(
            run(
                limit=args.limit,
                output_dir=args.output_dir,
                use_openai=args.use_openai,
                use_vllm=args.use_vllm,
                exclude_dirs=args.exclude_dir,
            )
        )

    elapsed = time.time() - start
    print(f"[build_dataset] done in {elapsed:.1f}s -> {result}")


if __name__ == "__main__":
    main()
