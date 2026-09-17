"""Push the intents10 bolAIndia_subset dataset to the Hugging Face Hub.

    .venv/bin/python -m pipeline.push_intents10 --dry-run --limit 5
        # builds a 5-row dataset locally and prints it - no network calls

    .venv/bin/python -m pipeline.push_intents10
        # incrementally append data.jsonl's new rows onto config.INTENTS10_HF_REPO

    .venv/bin/python -m pipeline.push_intents10 --full-rebuild
        # old, expensive approach: download the WHOLE existing dataset, merge
        # with local rows, de-dupe by id, re-upload everything. Only needed to
        # repair drift; normal runs should never need this.

Default (`append_incremental`) uploads ONLY the rows added locally since the
last push, as one additional parquet shard - it never re-downloads or
rewrites existing shards. Cost scales with the DELTA, not the whole dataset,
which is what makes this safe to call frequently even once the dataset is
large. It figures out how many rows are already on the Hub by reading the
repo's README `dataset_info` metadata (a few KB, not the actual data), then
bumps that same metadata after uploading the new shard so row/byte counts
stay accurate for `load_dataset`'s built-in verification.

If the repo/split doesn't exist yet, this just does a normal full push (which
creates the initial dataset_info) - every call after that is incremental.

pipeline/build_intents10.py calls `append_incremental()` periodically as it
runs; this module doubles as a manual/standalone way to push at any time.
"""

import argparse
import json
import os
import random
import re
import tempfile
import time
from datetime import datetime, timezone

import yaml
from datasets import Audio, Dataset, Features, Value, concatenate_datasets, load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from dataset_paths import resolve_chunk_path


def _now():
    return datetime.now(timezone.utc).isoformat()
from . import config
from .push_to_hub import _dedupe_keep_first

FEATURES = Features(
    {
        "id": Value("string"),
        "transcript": Value("string"),
        "start_ms": Value("float32"),
        "end_ms": Value("float32"),
        "duration_s": Value("float32"),
        "intent": Value("string"),
        "confidence": Value("float32"),
        "language_code": Value("string"),
        "channel": Value("string"),
        "conversation_id": Value("string"),
        "mongo_id": Value("string"),
        "source_db": Value("string"),
        # Native sample rate is preserved (source chunks are 16kHz mono PCM16).
        "audio": Audio(),
    }
)


def _iter_records(data_path, limit=None, audio_dir=None):
    n = 0
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            audio_filename = rec["audio"]
            audio_path = resolve_chunk_path(audio_filename, audio_dir)
            if not audio_path.exists():
                print(f"[push_intents10] WARNING: missing audio file, skipping: {audio_filename}")
                continue
            rec = {k: rec.get(k) for k in FEATURES if k != "audio"}
            rec["audio"] = str(audio_path)
            yield rec
            n += 1
            if limit is not None and n >= limit:
                return


def build_dataset(data_path, limit=None, audio_dir=None):
    # Dataset.from_generator proved non-deterministic here - repeated calls on
    # an unchanged file occasionally dropped trailing rows with no error (seen
    # dropping 2 of 10 in testing). Dataset.from_list on a fully-materialized
    # record list is reliable; rows only hold small metadata + a path string
    # (audio bytes aren't read until push time), so this stays cheap at scale.
    records = list(_iter_records(data_path, limit=limit, audio_dir=audio_dir))
    return Dataset.from_list(records, features=FEATURES)


def _split_ids(data_path, test_size, val_size, seed=42):
    """Shuffle all row ids (fixed seed, reproducible) then carve off
    test_size, then val_size, with everything else going to train."""
    ids = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["id"])

    rng = random.Random(seed)
    rng.shuffle(ids)

    total = len(ids)
    if test_size + val_size > total:
        raise ValueError(f"test_size ({test_size}) + val_size ({val_size}) exceeds total rows ({total})")

    test_ids = set(ids[:test_size])
    val_ids = set(ids[test_size : test_size + val_size])
    train_ids = set(ids[test_size + val_size :])
    return train_ids, val_ids, test_ids


def _iter_records_filtered(data_path, allowed_ids, audio_dir):
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["id"] not in allowed_ids:
                continue
            audio_filename = rec["audio"]
            audio_path = resolve_chunk_path(audio_filename, audio_dir)
            if not audio_path.exists():
                print(f"[push_intents10] WARNING: missing audio file, skipping: {audio_filename}")
                continue
            rec = {k: rec.get(k) for k in FEATURES if k != "audio"}
            rec["audio"] = str(audio_path)
            yield rec


def build_dataset_filtered(data_path, allowed_ids, audio_dir):
    records = list(_iter_records_filtered(data_path, allowed_ids, audio_dir))
    return Dataset.from_list(records, features=FEATURES)


def split_and_push(output_dir=None, repo_id=None, private=True, val_size=4000, test_size=1000, seed=42, token=None, audio_dir=None):
    """Carve data.jsonl into train/validation/test splits (fixed seed, so
    reproducible) and push each to its own split on repo_id - replacing
    whatever is currently in the "train" split with the smaller remainder.
    Built entirely from local data.jsonl + audio_dir; no download needed."""
    data_path, audio_dir = _resolve_paths(output_dir, audio_dir)
    repo_id = repo_id or config.INTENTS10_HF_REPO
    token = token or config.HF_TOKEN

    train_ids, val_ids, test_ids = _split_ids(data_path, test_size, val_size, seed=seed)
    print(
        f"[push_intents10] split sizes -> train={len(train_ids)} validation={len(val_ids)} test={len(test_ids)}"
    )

    counts = {}
    for split_name, allowed in (("test", test_ids), ("validation", val_ids), ("train", train_ids)):
        print(f"[push_intents10] building '{split_name}' ({len(allowed)} rows)...")
        ds = build_dataset_filtered(data_path, allowed, audio_dir)
        print(f"[push_intents10] pushing '{split_name}' ({len(ds)} rows) to {repo_id}...")
        ds.push_to_hub(repo_id, private=private, split=split_name, token=token)
        counts[split_name] = len(ds)

    _refresh_readme_body(repo_id, token)
    print(f"[push_intents10] done -> https://huggingface.co/datasets/{repo_id}")
    return counts


def _iter_records_from(data_path, start_index, audio_dir):
    """Like _iter_records, but skips the first start_index non-empty lines -
    those are the rows already on the Hub (data.jsonl is append-only, so a
    line count is a stable cursor)."""
    n = 0
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            if n <= start_index:
                continue
            rec = json.loads(line)
            audio_filename = rec["audio"]
            audio_path = resolve_chunk_path(audio_filename, audio_dir)
            if not audio_path.exists():
                print(f"[push_intents10] WARNING: missing audio file, skipping: {audio_filename}")
                continue
            rec = {k: rec.get(k) for k in FEATURES if k != "audio"}
            rec["audio"] = str(audio_path)
            yield rec


def build_delta_dataset(data_path, start_index, audio_dir):
    records = list(_iter_records_from(data_path, start_index, audio_dir))
    return Dataset.from_list(records, features=FEATURES) if records else None


def _read_hub_split_metadata(repo_id, split, token):
    """Read the repo's current row count for `split` from README's
    dataset_info YAML - a few KB, nowhere near the cost of downloading the
    actual data. Returns (num_examples, parsed_yaml_meta, readme_text_after_frontmatter).
    Raises EntryNotFoundError/RepositoryNotFoundError if the repo or README
    genuinely doesn't exist yet (the ONLY case that should fall back to a
    fresh push - any other error must propagate, since silently treating it
    as "doesn't exist" would push_to_hub a fresh split and delete the
    existing shards)."""
    readme_path = hf_hub_download(repo_id, "README.md", repo_type="dataset", token=token)
    content = open(readme_path, encoding="utf-8").read()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
    if not m:
        raise ValueError(f"{repo_id}'s README.md has no parseable YAML frontmatter")
    meta = yaml.safe_load(m.group(1))
    rest = m.group(2)
    for s in meta.get("dataset_info", {}).get("splits", []):
        if s["name"] == split:
            return s["num_examples"], meta, rest
    raise KeyError(f"split {split!r} not found in {repo_id}'s dataset_info metadata")


def _upload_delta_shard(repo_id, split, delta_ds, token):
    """Write delta_ds to a local parquet file and upload it as a NEW shard -
    existing shards are never downloaded or touched. Returns
    (path_in_repo, on_disk_bytes, in_memory_num_bytes)."""
    from datasets.table import embed_table_storage

    # Plain to_parquet() would serialize the Audio column as-is: a struct
    # {bytes: None, path: <local absolute path>} - a dangling reference no
    # one else can resolve. push_to_hub() embeds actual file bytes into the
    # table first (see Dataset._push_parquet_shards_to_hub_single); replicate
    # that exact step here.
    embedded = delta_ds.with_format("arrow").map(
        embed_table_storage, batched=True, batch_size=1000, keep_in_memory=True
    )
    embedded = embedded.with_format(None)

    tmp_path = tempfile.mktemp(suffix=".parquet")
    try:
        embedded.to_parquet(tmp_path)
        shard_bytes = os.path.getsize(tmp_path)
        # Timestamp-suffixed name avoids colliding with existing shards
        # (train-00000-of-NNNNN.parquet from a full push, or earlier deltas) -
        # datasets' hub loader matches any file under data/ starting with
        # "{split}-", it doesn't require the "-of-N" convention.
        shard_name = f"data/{split}-delta-{int(time.time() * 1000)}.parquet"
        HfApi(token=token).upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=shard_name,
            repo_id=repo_id,
            repo_type="dataset",
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return shard_name, shard_bytes, embedded.data.nbytes


def _bump_readme_counts(repo_id, split, meta, rest, delta_examples, delta_num_bytes, delta_download_size, token):
    """Update dataset_info's row/byte counts for `split` to include a
    just-uploaded delta shard, and re-upload README.md. Returns the split's
    new total num_examples."""
    new_total = None
    for s in meta["dataset_info"]["splits"]:
        if s["name"] == split:
            s["num_examples"] += delta_examples
            s["num_bytes"] += delta_num_bytes
            new_total = s["num_examples"]
    if new_total is None:
        raise KeyError(f"split {split!r} not found in {repo_id}'s dataset_info metadata")
    meta["dataset_info"]["download_size"] = meta["dataset_info"].get("download_size", 0) + delta_download_size
    meta["dataset_info"]["dataset_size"] = meta["dataset_info"].get("dataset_size", 0) + delta_num_bytes

    new_content = f"---\n{yaml.safe_dump(meta, sort_keys=False)}---\n{rest}"
    tmp_readme = tempfile.mktemp(suffix=".md")
    try:
        with open(tmp_readme, "w", encoding="utf-8") as f:
            f.write(new_content)
        HfApi(token=token).upload_file(
            path_or_fileobj=tmp_readme, path_in_repo="README.md", repo_id=repo_id, repo_type="dataset"
        )
    finally:
        if os.path.exists(tmp_readme):
            os.remove(tmp_readme)
    return new_total


def _all_split_counts(repo_id, token, splits=("train", "validation", "test")):
    counts = {}
    for s in splits:
        try:
            n, _, _ = _read_hub_split_metadata(repo_id, s, token)
            counts[s] = n
        except (EntryNotFoundError, RepositoryNotFoundError, DatasetNotFoundError, KeyError):
            pass
    return counts


def _build_readme_body(repo_id, token):
    """Descriptive markdown body appended after the YAML frontmatter -
    current per-split row counts and the live intent taxonomy, so the README
    stays accurate without anyone having to remember to update it by hand."""
    from .intents10 import INTENT_NAMES

    counts = _all_split_counts(repo_id, token)
    total = sum(counts.values())
    lines = [
        f"# {repo_id.split('/')[-1]}",
        "",
        "Per-utterance intent classification dataset built from `kapturecx/bolAIndia` "
        "(human-channel ASR chunks of Hindi/Hinglish customer calls), labeled with "
        f"Gemini ({config.GEMINI_MODEL}) against a closed taxonomy "
        f"(`{os.path.basename(config.INTENTS10_TAXONOMY_PATH)}`, {len(INTENT_NAMES)} intents).",
        "",
        f"**Last updated:** {_now()}",
        "",
        "## Splits",
        "",
        "| split | rows |",
        "|---|---|",
    ]
    for s in ("train", "validation", "test"):
        if s in counts:
            lines.append(f"| {s} | {counts[s]:,} |")
    lines += [
        f"| **total** | **{total:,}** |",
        "",
        "## Fields",
        "",
        "| Field | Description |",
        "|---|---|",
        "| `id` | stable unique row id (source `chunk_id`) |",
        "| `audio` | audio clip for this turn (native sample rate, no forced resampling) |",
        "| `transcript` | ASR transcript |",
        "| `start_ms` / `end_ms` | chunk position within the source recording (ms) |",
        "| `duration_s` | chunk duration in seconds |",
        "| `intent` | one of the closed-set intents below |",
        "| `confidence` / `language_code` / `channel` | ASR provider metadata |",
        "| `conversation_id` / `mongo_id` / `source_db` | source call provenance |",
        "",
        "## Intents",
        "",
        ", ".join(f"`{n}`" for n in INTENT_NAMES),
        "",
    ]
    return "\n".join(lines)


def _refresh_readme_body(repo_id, token):
    """Re-fetch the CURRENT frontmatter (never trust a possibly-stale local
    copy) and replace only the body below it with fresh stats - called after
    every push so the README never drifts from what's actually on the Hub."""
    readme_path = hf_hub_download(repo_id, "README.md", repo_type="dataset", token=token)
    content = open(readme_path, encoding="utf-8").read()
    m = re.match(r"^(---\n.*?\n---\n)", content, re.DOTALL)
    frontmatter = m.group(1) if m else "---\n---\n"

    new_content = frontmatter + "\n" + _build_readme_body(repo_id, token)
    tmp_readme = tempfile.mktemp(suffix=".md")
    try:
        with open(tmp_readme, "w", encoding="utf-8") as f:
            f.write(new_content)
        HfApi(token=token).upload_file(
            path_or_fileobj=tmp_readme, path_in_repo="README.md", repo_id=repo_id, repo_type="dataset"
        )
    finally:
        if os.path.exists(tmp_readme):
            os.remove(tmp_readme)


def _resolve_paths(output_dir, audio_dir):
    output_dir = output_dir or config.INTENTS10_OUTPUT_DIR
    audio_dir = audio_dir or os.path.join(output_dir, config.INTENTS10_AUDIO_DIRNAME)
    data_path = os.path.join(output_dir, config.INTENTS10_DATA_FILENAME)
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    return data_path, audio_dir


def push(output_dir=None, repo_id=None, private=True, split="train", limit=None, token=None, audio_dir=None):
    data_path, audio_dir = _resolve_paths(output_dir, audio_dir)
    repo_id = repo_id or config.INTENTS10_HF_REPO
    token = token or config.HF_TOKEN

    print(f"[push_intents10] building dataset from {data_path}" + (f" (limit={limit})" if limit else ""))
    ds = build_dataset(data_path, limit=limit, audio_dir=audio_dir)
    print(f"[push_intents10] {len(ds)} rows -> pushing to {repo_id} (private={private}, split={split})...")
    ds.push_to_hub(repo_id, private=private, split=split, token=token)
    _refresh_readme_body(repo_id, token)
    print(f"[push_intents10] done -> https://huggingface.co/datasets/{repo_id}")
    return {"repo_id": repo_id, "total_rows": len(ds)}


def append(output_dir=None, repo_id=None, private=True, split="train", limit=None, token=None, audio_dir=None):
    """Append local data.jsonl onto whatever is already on the Hub (or push
    fresh if the repo/split doesn't exist yet). De-dupes by `id`, so safe to
    call repeatedly as more rows accumulate locally."""
    data_path, audio_dir = _resolve_paths(output_dir, audio_dir)
    repo_id = repo_id or config.INTENTS10_HF_REPO
    token = token or config.HF_TOKEN

    new_ds = build_dataset(data_path, limit=limit, audio_dir=audio_dir)

    try:
        existing_ds = load_dataset(repo_id, split=split, token=token)
        print(f"[push_intents10] existing on Hub: {len(existing_ds)} rows")
        combined = concatenate_datasets([existing_ds, new_ds])
        combined = _dedupe_keep_first(combined, key="id")
    except DatasetNotFoundError as e:
        # Only a genuinely missing repo/split means "push fresh" - anything
        # else (disk full downloading the existing copy, network error, auth)
        # must propagate so it's retried later instead of silently pushing
        # `new_ds` alone as if nothing existed on the Hub yet.
        print(f"[push_intents10] no existing '{split}' split found ({type(e).__name__}: {e}) - pushing fresh")
        combined = new_ds

    print(f"[push_intents10] pushing {len(combined)} total rows to {repo_id} (split={split})...")
    combined.push_to_hub(repo_id, private=private, split=split, token=token)
    _refresh_readme_body(repo_id, token)
    print(f"[push_intents10] done -> https://huggingface.co/datasets/{repo_id}")
    return {"repo_id": repo_id, "total_rows": len(combined)}


def append_incremental(
    output_dir=None, repo_id=None, private=True, split="train", token=None, audio_dir=None, local_offset=0
):
    """Upload only the LOCAL rows added since the last push, as a new parquet
    shard - existing shards are never downloaded or rewritten. Cost scales
    with the delta, not the whole dataset. Falls back to a normal fresh push
    only when the repo/README genuinely doesn't exist yet.

    local_offset: the Hub's row count for `split` is used as a cursor into
    data.jsonl - this only works while that count equals a clean PREFIX of
    the local file. If rows were ever surgically removed from the middle of
    this split (e.g. pipeline/push_intents10.py's shard-extraction to build
    validation/test - see data.jsonl.split_offset.json), that assumption
    breaks: the Hub count under-reports how far into data.jsonl is already
    accounted for. local_offset corrects for that - pass the total rows
    removed that way so far, and this adds it back before slicing."""
    data_path, audio_dir = _resolve_paths(output_dir, audio_dir)
    repo_id = repo_id or config.INTENTS10_HF_REPO
    token = token or config.HF_TOKEN

    try:
        start_index, meta, rest = _read_hub_split_metadata(repo_id, split, token)
    except (EntryNotFoundError, RepositoryNotFoundError, DatasetNotFoundError) as e:
        print(f"[push_intents10] no existing '{split}' metadata found ({type(e).__name__}: {e}) - pushing fresh")
        return push(output_dir=output_dir, repo_id=repo_id, private=private, split=split, audio_dir=audio_dir, token=token)

    if local_offset:
        print(f"[push_intents10] applying local_offset={local_offset} (rows previously moved out of '{split}')")
    start_index += local_offset

    delta_ds = build_delta_dataset(data_path, start_index, audio_dir)
    if delta_ds is None or len(delta_ds) == 0:
        print(f"[push_intents10] nothing new to push ({start_index} rows already accounted for)")
        return {"repo_id": repo_id, "total_rows": start_index - local_offset}

    print(f"[push_intents10] {start_index} local rows already accounted for - uploading {len(delta_ds)} new rows as a delta shard...")
    shard_name, shard_bytes, shard_num_bytes = _upload_delta_shard(repo_id, split, delta_ds, token)
    print(f"[push_intents10] uploaded {shard_name} ({shard_bytes} bytes)")
    new_total = _bump_readme_counts(repo_id, split, meta, rest, len(delta_ds), shard_num_bytes, shard_bytes, token)
    _refresh_readme_body(repo_id, token)
    print(f"[push_intents10] done -> https://huggingface.co/datasets/{repo_id} (total rows: {new_total})")
    return {"repo_id": repo_id, "total_rows": new_total}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=None, help=f"default: {config.INTENTS10_OUTPUT_DIR}")
    parser.add_argument("--audio-dir", default=None, help="default: <output-dir>/audio")
    parser.add_argument("--repo-id", default=None, help=f"default: {config.INTENTS10_HF_REPO}")
    parser.add_argument("--split", default="train")
    parser.add_argument("--public", action="store_true", help="push as a public dataset (default: private)")
    parser.add_argument("--limit", type=int, default=None, help="only applies to --fresh / --full-rebuild")
    parser.add_argument("--fresh", action="store_true", help="push fresh (all local rows), ignoring anything already on the Hub")
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help=(
            "old, expensive path: download the whole existing dataset, merge with "
            "local rows, de-dupe by id, re-upload everything. Only for repairing "
            "drift - normal use should never need this."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="build the dataset locally and print a sample - no network calls")
    parser.add_argument(
        "--make-splits",
        action="store_true",
        help="carve data.jsonl into train/validation/test splits and push each (see --val-size/--test-size/--seed)",
    )
    parser.add_argument("--val-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dry_run:
        data_path, audio_dir = _resolve_paths(args.output_dir, args.audio_dir)
        ds = build_dataset(data_path, limit=args.limit or 5, audio_dir=audio_dir)
        print(ds)
        # decode=False: printing a sample row should work even when the local
        # torchcodec/ffmpeg versions can't actually decode audio (push_to_hub
        # itself never needs to decode - it just uploads the raw bytes/path).
        print("\nsample row 0 (audio undecoded):")
        print(ds.cast_column("audio", Audio(decode=False))[0])
        return

    if args.make_splits:
        result = split_and_push(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            private=not args.public,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
            audio_dir=args.audio_dir,
        )
        print(f"[push_intents10] result -> {result}")
        return

    kwargs = dict(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        private=not args.public,
        split=args.split,
        audio_dir=args.audio_dir,
    )
    if args.fresh:
        result = push(limit=args.limit, **kwargs)
    elif args.full_rebuild:
        result = append(limit=args.limit, **kwargs)
    else:
        result = append_incremental(**kwargs)
    print(f"[push_intents10] result -> {result}")


if __name__ == "__main__":
    main()
