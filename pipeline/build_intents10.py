"""Build the intents10 bolAIndia_subset dataset: intent label per ASR chunk
from kapturecx/bolAIndia, audio saved locally, resumable and pushed to the Hub.

The source dataset already carries a provider transcript per chunk (no
transcription step needed here, unlike pipeline/build_dataset.py) - this
script only saves each chunk's audio locally and classifies its intent with
Gemini using the taxonomy at config.INTENTS10_TAXONOMY_PATH (currently
intents15.json - see pipeline/relabel_intents15.py to relabel rows already
classified under an earlier version of the taxonomy).

Run with the venv's python as a module:

    .venv/bin/python -m pipeline.build_intents10 sample --n 20   # quick check, separate dir
    .venv/bin/python -m pipeline.build_intents10 run              # full run, resumable
    .venv/bin/python -m pipeline.build_intents10 run --limit 5000 # cap NEW rows this call
    .venv/bin/python -m pipeline.build_intents10 run --no-push    # classify/save only, push later by hand

Resuming (Ctrl-C / cancel at any point, then re-run `run`):
  - Every finalized row is appended to data.jsonl immediately - chunk_ids
    already there (or still buffered in pending_intent.jsonl awaiting
    classification) are skipped on the next run.
  - The source is a HF streaming dataset; its exact iteration position is
    checkpointed to stream_state.json (datasets' IterableDataset state_dict)
    every --fetch-batch-size rows, so a resumed run picks up near where it
    left off instead of re-reading everything before it.
  - A small fraction of rows appear twice in the source (two workers
    overlapped before their work was partitioned) - `chunk_id` is the row key,
    de-duped in-memory via `seen_ids` for the life of the run.

Intent classification is batched (config.INTENT_BATCH_SIZE transcripts per
Gemini call, buffered in pending_intent.jsonl so a stop mid-batch loses no
already-saved audio/transcript work). Up to --classify-concurrency of those
batch calls run concurrently in a thread pool (default: config value, see
INTENTS10_CLASSIFY_MAX_CONCURRENCY) instead of one at a time - this is the
main throughput lever, since each call is dominated by Gemini round-trip
latency, not local CPU. A classification failure (quota, auth, unexpected
exception, after retries) STOPS the run; progress already written is safe,
just re-run `run` to resume.

Pushing to the Hub (kapturecx/S2I-10-v1 by default) is kicked off in a
background thread once every --push-every-rows NEW finalized rows (so it
doesn't stall fetching/classifying), via pipeline/push_intents10.py's
`append_incremental()` - plus a final, synchronous push at the end of the
run. That uploads only the new rows as an additional parquet shard (existing
shards are never re-downloaded or rewritten), so cost scales with the delta,
not the whole dataset - safe to call frequently even once the dataset is
large. Use --no-push and run `python -m pipeline.push_intents10` by hand for
full control.
"""

import argparse
import io
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import soundfile as sf
from datasets import Audio, load_dataset

from . import config
from .intents10 import build_classifier


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    continue
    return ids


def _count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_processed_ids(data_path, pending_path):
    return _read_ids(data_path) | _read_ids(pending_path)


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


def _rewrite_pending_file(pending_path, pending_buffer):
    tmp_path = pending_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in pending_buffer:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, pending_path)


def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path, obj):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp_path, path)


def _save_audio(raw_bytes, out_path):
    """Decode whatever container the source bytes are in and write a plain
    PCM16 wav - guarantees valid, uniform audio files regardless of how the
    source stored them. Returns duration in seconds."""
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="int16", always_2d=False)
    sf.write(out_path, data, sr, subtype="PCM_16")
    return len(data) / float(sr)


def _classify_one_batch(classifier, transcripts):
    """Single classify_batch call with retries, run from a worker thread - no
    file I/O here so concurrent calls need no locking. Returns
    (intents_or_None, stop_reason, traceback_str)."""
    last_error = None
    last_traceback = None
    total_attempts = 1 + config.CLASSIFY_MAX_RETRIES
    for attempt in range(1, total_attempts + 1):
        try:
            return classifier.classify_batch(transcripts), None, None
        except Exception as e:
            last_error = e
            last_traceback = traceback.format_exc()
            if attempt < total_attempts:
                print(
                    f"[build_intents10] batch classification attempt {attempt}/{total_attempts} "
                    f"failed ({type(e).__name__}: {e}), retrying in {config.CLASSIFY_RETRY_DELAY_S}s..."
                )
                time.sleep(config.CLASSIFY_RETRY_DELAY_S)

    stop_reason = f"{type(last_error).__name__}: {last_error}"
    return None, stop_reason, last_traceback


def _flush_pending(classifier, pending_buffer, data_f, err_f, pending_path, max_concurrency=1):
    """Classify buffered transcripts in sub-batches of config.INTENT_BATCH_SIZE,
    up to max_concurrency of them in flight at once (each is dominated by
    Gemini round-trip latency, so this is the main throughput lever).

    Sub-batches are taken from the FRONT of pending_buffer each round. If one
    fails (after retries), everything before it in this round is written and
    dropped from pending_buffer as usual, but anything from/after the failed
    one is discarded from this round's results (even if it happened to
    succeed) so pending_buffer stays a clean, contiguous remaining suffix -
    then the run stops, same as the sequential (max_concurrency=1) case.
    Returns (ok, stop_reason, n_flushed)."""
    total_flushed = 0
    while pending_buffer:
        sub_batches = []
        offset = 0
        while offset < len(pending_buffer) and len(sub_batches) < max_concurrency:
            sub_batches.append(pending_buffer[offset : offset + config.INTENT_BATCH_SIZE])
            offset += config.INTENT_BATCH_SIZE

        results = [None] * len(sub_batches)
        if len(sub_batches) == 1:
            results[0] = _classify_one_batch(classifier, [r["transcript"] for r in sub_batches[0]])
        else:
            with ThreadPoolExecutor(max_workers=len(sub_batches)) as ex:
                future_to_idx = {
                    ex.submit(_classify_one_batch, classifier, [r["transcript"] for r in b]): i
                    for i, b in enumerate(sub_batches)
                }
                for fut in as_completed(future_to_idx):
                    results[future_to_idx[fut]] = fut.result()

        consumed = 0
        stop_reason = None
        for b, (intents, reason, tb) in zip(sub_batches, results):
            if intents is None:
                stop_reason = reason
                err_f.write(
                    json.dumps(
                        {
                            "batch_size": len(b),
                            "error": reason,
                            "attempts": 1 + config.CLASSIFY_MAX_RETRIES,
                            "traceback": tb,
                            "time": _now(),
                        }
                    )
                    + "\n"
                )
                err_f.flush()
                break
            for rec, intent in zip(b, intents):
                final_record = dict(rec)
                final_record["intent"] = intent
                data_f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
            consumed += len(b)

        data_f.flush()
        total_flushed += consumed
        if consumed:
            print(f"[build_intents10] classified {consumed} ({len(sub_batches)} batch(es) this round)")
            del pending_buffer[:consumed]
            _rewrite_pending_file(pending_path, pending_buffer)

        if stop_reason is not None:
            print(f"[build_intents10] STOPPED on batch classification error: {stop_reason}")
            print(f"[build_intents10] Progress saved (transcripts kept in {os.path.basename(pending_path)}). Re-run to resume.")
            return False, stop_reason, total_flushed

    return True, None, total_flushed


class _PushState:
    """Runs pipeline/push_intents10.py's append() in a background thread so it
    doesn't block fetching/classifying. Only one push runs at a time - a
    checkpoint that lands while one is still in flight just skips starting a
    new one and tries again at the next checkpoint."""

    def __init__(self, rows_pushed, output_dir, audio_dir, push_state_path):
        self.rows_pushed = rows_pushed
        self.output_dir = output_dir
        self.audio_dir = audio_dir
        self.push_state_path = push_state_path
        self._lock = threading.Lock()
        self._thread = None

    def is_busy(self):
        return self._thread is not None and self._thread.is_alive()

    def maybe_start(self, total_finalized, push_every):
        if self.is_busy() or (total_finalized - self.rows_pushed) < push_every:
            return
        print(f"[build_intents10] starting background push ({total_finalized - self.rows_pushed} new rows)...")

        def _worker():
            pushed = _try_push(self.output_dir, self.audio_dir, self.push_state_path)
            if pushed is not None:
                with self._lock:
                    self.rows_pushed = pushed

        self._thread = threading.Thread(target=_worker, name="intents10-push")
        self._thread.start()

    def join(self):
        if self._thread is not None:
            self._thread.join()


def _paths(output_dir):
    audio_dir = os.path.join(output_dir, config.INTENTS10_AUDIO_DIRNAME)
    return {
        "audio_dir": audio_dir,
        "data": os.path.join(output_dir, config.INTENTS10_DATA_FILENAME),
        "pending": os.path.join(output_dir, config.INTENTS10_PENDING_FILENAME),
        "errors": os.path.join(output_dir, config.INTENTS10_ERRORS_FILENAME),
        "stream_state": os.path.join(output_dir, config.INTENTS10_STREAM_STATE_FILENAME),
        "push_state": os.path.join(output_dir, config.INTENTS10_PUSH_STATE_FILENAME),
    }


def _try_push(output_dir, audio_dir, push_state_path):
    from . import push_intents10

    # train_local_offset: rows surgically moved OUT of the "train" split
    # (e.g. shard extraction to build validation/test - see
    # pipeline/push_intents10.py's append_incremental docstring). Without it,
    # the Hub's train row count under-counts how far into data.jsonl is
    # already accounted for, and the next incremental push would re-add
    # already-distributed rows as if they were new. Persisted here so it
    # survives across runs; carried forward on every save.
    push_state = _load_json(push_state_path, {})
    local_offset = push_state.get("train_local_offset", 0)

    try:
        result = push_intents10.append_incremental(
            output_dir=output_dir, audio_dir=audio_dir, local_offset=local_offset
        )
        _save_json(
            push_state_path,
            {"total_rows_pushed": result["total_rows"], "train_local_offset": local_offset, "time": _now()},
        )
        print(f"[build_intents10] pushed -> {result}")
        return result["total_rows"]
    except Exception as e:
        print(f"[build_intents10] push failed (will retry at the next checkpoint): {type(e).__name__}: {e}")
        return None


def run(
    limit=None,
    output_dir=None,
    reset=False,
    fetch_batch_size=None,
    push_every=None,
    do_push=True,
    classify_concurrency=None,
):
    output_dir = output_dir or config.INTENTS10_OUTPUT_DIR
    p = _paths(output_dir)
    os.makedirs(p["audio_dir"], exist_ok=True)

    if reset:
        for key in ("data", "pending", "errors", "stream_state", "push_state"):
            if os.path.exists(p[key]):
                os.remove(p[key])

    fetch_batch_size = fetch_batch_size or config.INTENTS10_FETCH_BATCH_SIZE
    push_every = config.INTENTS10_PUSH_EVERY_ROWS if push_every is None else push_every
    classify_concurrency = classify_concurrency or config.INTENTS10_CLASSIFY_MAX_CONCURRENCY

    seen_ids = load_processed_ids(p["data"], p["pending"])
    num_processed_prior = _count_lines(p["data"])
    pending_buffer = _load_pending(p["pending"])
    if pending_buffer:
        print(f"[build_intents10] {len(pending_buffer)} chunks already saved and awaiting classification")
    print(f"[build_intents10] {len(seen_ids)} chunk_ids already done/pending, {num_processed_prior} finalized rows")

    push_state = _load_json(p["push_state"], {"total_rows_pushed": 0})
    rows_pushed = push_state.get("total_rows_pushed", 0)
    pusher = _PushState(rows_pushed, output_dir, p["audio_dir"], p["push_state"])

    ds = load_dataset(
        config.INTENTS10_SOURCE_REPO,
        config.INTENTS10_SOURCE_CONFIG,
        split=config.INTENTS10_SOURCE_SPLIT,
        streaming=True,
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    stream_state = _load_json(p["stream_state"])
    if stream_state is not None:
        ds.load_state_dict(stream_state)
        print(f"[build_intents10] resumed source stream position from {p['stream_state']}")

    classifier = build_classifier()
    print(f"[build_intents10] intent backend: {classifier.name} ({classifier.model})")

    processed_this_run = 0
    fetched_this_run = 0
    stopped = False
    stop_reason = None

    with open(p["data"], "a", encoding="utf-8") as data_f, open(p["errors"], "a", encoding="utf-8") as err_f:
        for row in ds:
            chunk_id = row["chunk_id"]
            if chunk_id in seen_ids:
                continue
            if limit is not None and fetched_this_run >= limit:
                break

            fetched_this_run += 1
            seen_ids.add(chunk_id)

            audio_filename = f"{chunk_id}.wav"
            audio_path = os.path.join(p["audio_dir"], audio_filename)
            try:
                duration_s = _save_audio(row["audio"]["bytes"], audio_path)
            except Exception as e:
                err_f.write(
                    json.dumps({"chunk_id": chunk_id, "stage": "audio_decode", "error": str(e), "time": _now()})
                    + "\n"
                )
                err_f.flush()
                print(f"[build_intents10] SKIPPED (audio decode failed) {chunk_id}: {e}")
                continue

            transcript = (row.get("text") or "").strip()
            record_partial = {
                "id": chunk_id,
                "audio": audio_filename,
                "transcript": transcript,
                "start_ms": row.get("chunk_start_ms"),
                "end_ms": row.get("chunk_end_ms"),
                "duration_s": round(duration_s, 3),
                "confidence": row.get("confidence"),
                "language_code": row.get("language_code"),
                "channel": row.get("channel"),
                "conversation_id": row.get("conversation_id"),
                "mongo_id": row.get("mongo_id"),
                "source_db": row.get("source_db"),
            }

            if not transcript or row.get("is_unintelligible"):
                # No need to spend an LLM call - empty/unintelligible transcript is noise.
                record_partial["intent"] = "BACKCHANNEL_OR_NOISE"
                data_f.write(json.dumps(record_partial, ensure_ascii=False) + "\n")
                data_f.flush()
                processed_this_run += 1
            else:
                pending_buffer.append(record_partial)
                with open(p["pending"], "a", encoding="utf-8") as pf:
                    pf.write(json.dumps(record_partial, ensure_ascii=False) + "\n")

                # Wait for enough to fill the whole concurrency pool at once -
                # flushing right at INTENT_BATCH_SIZE would only ever form one
                # sub-batch, leaving classify_concurrency mostly unused.
                if len(pending_buffer) >= config.INTENT_BATCH_SIZE * classify_concurrency:
                    ok, reason, n = _flush_pending(
                        classifier, pending_buffer, data_f, err_f, p["pending"], max_concurrency=classify_concurrency
                    )
                    processed_this_run += n
                    if not ok:
                        stop_reason = reason
                        stopped = True
                        break

            if fetched_this_run % fetch_batch_size == 0:
                stream_state = ds.state_dict()
                _save_json(p["stream_state"], stream_state)
                total_finalized = num_processed_prior + processed_this_run
                print(
                    f"[build_intents10] checkpoint: {fetched_this_run} fetched this run, "
                    f"{total_finalized} finalized total, {len(pending_buffer)} pending"
                )

                if do_push:
                    pusher.maybe_start(total_finalized, push_every)

        if pending_buffer and not stopped:
            ok, reason, n = _flush_pending(
                classifier, pending_buffer, data_f, err_f, p["pending"], max_concurrency=classify_concurrency
            )
            processed_this_run += n
            if not ok:
                stop_reason = reason
                stopped = True

    try:
        _save_json(p["stream_state"], ds.state_dict())
    except Exception:
        pass

    total_finalized = num_processed_prior + processed_this_run
    if do_push:
        pusher.join()  # let any in-flight background push land before the final one
        if total_finalized > pusher.rows_pushed:
            pushed = _try_push(output_dir, p["audio_dir"], p["push_state"])
            if pushed is not None:
                pusher.rows_pushed = pushed

    return {
        "fetched_this_run": fetched_this_run,
        "processed_this_run": processed_this_run,
        "total_finalized": total_finalized,
        "pending_unclassified": len(pending_buffer),
        "rows_pushed": pusher.rows_pushed,
        "stopped": stopped,
        "stop_reason": stop_reason,
    }


def run_sample(n=20, output_dir=None):
    """Verification helper: (re)generate just the first `n` new chunks into a
    separate sample dir. Safe to call repeatedly - starts fresh each time and
    never touches the main resumable dataset dir."""
    output_dir = output_dir or os.path.join(config.DATA_DIR, "intents10", "bolAIndia_subset_sample")
    return run(limit=n, output_dir=output_dir, reset=True, do_push=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sample_p = sub.add_parser("sample", help="Run a quick sample for verification (no push)")
    sample_p.add_argument("--n", type=int, default=20)

    run_p = sub.add_parser("run", help="Run/resume the full generation")
    run_p.add_argument("--limit", type=int, default=None, help="Cap on number of NEW chunks to process this call")
    run_p.add_argument("--output-dir", type=str, default=config.INTENTS10_OUTPUT_DIR)
    run_p.add_argument("--fetch-batch-size", type=int, default=None)
    run_p.add_argument("--push-every-rows", type=int, default=None)
    run_p.add_argument(
        "--classify-concurrency",
        type=int,
        default=None,
        help=f"concurrent Gemini classify_batch calls in flight (default: {config.INTENTS10_CLASSIFY_MAX_CONCURRENCY})",
    )
    run_p.add_argument("--no-push", action="store_true", help="classify/save only - push later with pipeline.push_intents10")
    run_p.add_argument("--reset", action="store_true", help="wipe local progress in --output-dir and start over")

    args = parser.parse_args()

    start = time.time()
    if args.command == "sample":
        result = run_sample(n=args.n)
    else:
        result = run(
            limit=args.limit,
            output_dir=args.output_dir,
            reset=args.reset,
            fetch_batch_size=args.fetch_batch_size,
            push_every=args.push_every_rows,
            do_push=not args.no_push,
            classify_concurrency=args.classify_concurrency,
        )

    elapsed = time.time() - start
    print(f"[build_intents10] done in {elapsed:.1f}s -> {result}")


if __name__ == "__main__":
    main()
