"""Relabel the existing intents10 bolAIndia_subset dataset with the updated
intents15.json taxonomy (added GREETING/VOICE_MAIL, split ACTION_COMMITMENT
into ACTION_COMMITMENT_NOW/_LATER - see pipeline/config.py's
INTENTS10_TAXONOMY_PATH).

Re-classifies every row's transcript from scratch against the new taxonomy -
old intent labels are replaced entirely, not merged/patched.

Safe to run alongside the live build_intents10.py pipeline: this script only
READS data.jsonl and writes to its own separate relabel_output.jsonl - it
never touches data.jsonl until you explicitly run `finalize`.

    .venv/bin/python -m pipeline.relabel_intents15 relabel              # run/resume relabeling
    .venv/bin/python -m pipeline.relabel_intents15 relabel --limit 5000  # cap this call
    .venv/bin/python -m pipeline.relabel_intents15 status               # progress check
    .venv/bin/python -m pipeline.relabel_intents15 finalize             # swap corrected labels into data.jsonl

`finalize`:
  - Refuses unless EVERY row currently in data.jsonl has a relabeled
    counterpart (run `relabel` again to catch up if the live pipeline added
    more rows since you started, or if relabeling isn't 100% done yet).
  - Backs up the original to data.jsonl.pre_v15.bak, then atomically replaces
    data.jsonl with the corrected version (original row order preserved).
  - Does NOT push to the Hub. Every row's label may have changed, so this
    needs a full push (not the usual incremental append) - run afterward:
        .venv/bin/python -m pipeline.push_intents10 --fresh
    That re-embeds and re-uploads all audio once (built from local files, no
    download of the existing Hub copy needed) - a one-time cost, not a
    recurring one.
"""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .build_intents10 import _classify_one_batch, _now
from .intents10 import build_classifier


def _paths(output_dir):
    return {
        "data": os.path.join(output_dir, config.INTENTS10_DATA_FILENAME),
        "output": os.path.join(output_dir, "relabel_output.jsonl"),
        "errors": os.path.join(output_dir, "relabel_errors.log"),
        "backup": os.path.join(output_dir, "data.jsonl.pre_v15.bak"),
    }


def _count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _read_relabeled_ids(output_path):
    ids = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    continue
    return ids


def _iter_pending_rows(data_path, done_ids):
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["id"] in done_ids:
                continue
            yield rec


def relabel(output_dir=None, limit=None, batch_size=None, concurrency=None):
    output_dir = output_dir or config.INTENTS10_OUTPUT_DIR
    p = _paths(output_dir)
    batch_size = batch_size or config.INTENT_BATCH_SIZE
    concurrency = concurrency or config.INTENTS10_CLASSIFY_MAX_CONCURRENCY

    done_ids = _read_relabeled_ids(p["output"])
    total_lines = _count_lines(p["data"])
    print(f"[relabel_intents15] {len(done_ids)}/{total_lines} rows already relabeled")

    classifier = build_classifier()
    print(f"[relabel_intents15] backend: {classifier.name} ({classifier.model})")

    processed_this_run = 0
    stopped = False
    stop_reason = None

    with open(p["output"], "a", encoding="utf-8") as out_f, open(p["errors"], "a", encoding="utf-8") as err_f:

        def flush(pending_batch):
            nonlocal stopped, stop_reason, processed_this_run
            sub_batches = [pending_batch[i : i + batch_size] for i in range(0, len(pending_batch), batch_size)]
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

            for b, (intents, reason, tb) in zip(sub_batches, results):
                if intents is None:
                    stop_reason = reason
                    err_f.write(
                        json.dumps({"batch_size": len(b), "error": reason, "traceback": tb, "time": _now()}) + "\n"
                    )
                    err_f.flush()
                    stopped = True
                    break
                for rec, intent in zip(b, intents):
                    final = dict(rec)
                    final["intent"] = intent
                    out_f.write(json.dumps(final, ensure_ascii=False) + "\n")
                processed_this_run += len(b)
            out_f.flush()

        pending_batch = []
        for rec in _iter_pending_rows(p["data"], done_ids):
            if limit is not None and processed_this_run >= limit:
                break

            transcript = (rec.get("transcript") or "").strip()
            if not transcript:
                # Same shortcut the live pipeline uses - GREETING/VOICE_MAIL/etc
                # are all about SPEECH content, an empty transcript is still noise.
                final = dict(rec)
                final["intent"] = "BACKCHANNEL_OR_NOISE"
                out_f.write(json.dumps(final, ensure_ascii=False) + "\n")
                out_f.flush()
                processed_this_run += 1
                continue

            pending_batch.append(rec)
            if len(pending_batch) >= batch_size * concurrency:
                flush(pending_batch)
                pending_batch = []
                if stopped:
                    break

            if processed_this_run and processed_this_run % 2000 == 0:
                print(
                    f"[relabel_intents15] {processed_this_run} relabeled this run "
                    f"({len(done_ids) + processed_this_run}/{total_lines} total)"
                )

        if pending_batch and not stopped:
            flush(pending_batch)

    total_done = len(_read_relabeled_ids(p["output"]))
    print(f"[relabel_intents15] done this run: {processed_this_run}, total relabeled: {total_done}/{total_lines}")
    if stopped:
        print(f"[relabel_intents15] STOPPED: {stop_reason} - re-run to resume")
    return {
        "processed_this_run": processed_this_run,
        "total_relabeled": total_done,
        "total_rows": total_lines,
        "stopped": stopped,
        "stop_reason": stop_reason,
    }


def status(output_dir=None):
    output_dir = output_dir or config.INTENTS10_OUTPUT_DIR
    p = _paths(output_dir)
    total = _count_lines(p["data"])
    done = len(_read_relabeled_ids(p["output"]))
    print(f"[relabel_intents15] {done}/{total} relabeled ({total - done} remaining)")
    return {"done": done, "total": total}


def finalize(output_dir=None):
    output_dir = output_dir or config.INTENTS10_OUTPUT_DIR
    p = _paths(output_dir)

    relabeled = {}
    with open(p["output"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                relabeled[rec["id"]] = rec

    # Verify EVERY row currently in data.jsonl has a relabeled counterpart -
    # refuse otherwise (e.g. the live pipeline appended more rows since
    # relabeling started, or relabeling just isn't finished yet).
    missing = []
    ordered_ids = []
    with open(p["data"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec_id = json.loads(line)["id"]
            ordered_ids.append(rec_id)
            if rec_id not in relabeled:
                missing.append(rec_id)

    if missing:
        raise RuntimeError(
            f"{len(missing)} rows in data.jsonl have no relabeled counterpart yet "
            f"(e.g. {missing[:5]}) - run `relabel` again (check progress with `status`) "
            "until it's fully caught up before finalizing."
        )

    shutil.copy2(p["data"], p["backup"])
    print(f"[relabel_intents15] backed up original to {p['backup']}")

    tmp_path = p["data"] + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec_id in ordered_ids:
            f.write(json.dumps(relabeled[rec_id], ensure_ascii=False) + "\n")
    os.replace(tmp_path, p["data"])

    print(f"[relabel_intents15] replaced data.jsonl with {len(ordered_ids)} relabeled rows")
    print(
        "[relabel_intents15] next: push the corrected dataset with "
        "`python -m pipeline.push_intents10 --fresh` (a full push - every "
        "row's label may have changed)"
    )
    return {"total_rows": len(ordered_ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    relabel_p = sub.add_parser("relabel", help="Run/resume relabeling against the current taxonomy")
    relabel_p.add_argument("--output-dir", default=config.INTENTS10_OUTPUT_DIR)
    relabel_p.add_argument("--limit", type=int, default=None, help="cap rows relabeled this call")
    relabel_p.add_argument("--batch-size", type=int, default=None)
    relabel_p.add_argument("--concurrency", type=int, default=None)

    status_p = sub.add_parser("status", help="Show relabeling progress")
    status_p.add_argument("--output-dir", default=config.INTENTS10_OUTPUT_DIR)

    finalize_p = sub.add_parser("finalize", help="Swap corrected labels into data.jsonl (only once 100% relabeled)")
    finalize_p.add_argument("--output-dir", default=config.INTENTS10_OUTPUT_DIR)

    args = parser.parse_args()
    start = time.time()
    if args.command == "relabel":
        result = relabel(
            output_dir=args.output_dir, limit=args.limit, batch_size=args.batch_size, concurrency=args.concurrency
        )
    elif args.command == "status":
        result = status(output_dir=args.output_dir)
    else:
        result = finalize(output_dir=args.output_dir)

    elapsed = time.time() - start
    print(f"[relabel_intents15] done in {elapsed:.1f}s -> {result}")


if __name__ == "__main__":
    main()
