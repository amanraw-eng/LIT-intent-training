"""Push the generated call_trascript_intent_data dataset to the Hugging Face Hub.

Builds a HF `datasets.Dataset` with an `audio` column (decoded from each
chunk's local wav file), a stable `id`, `transcript`, `intent`, plus the rest
of data.jsonl's fields for traceability (oid, conversation_id, recording_url,
chunk_index, start_ms, end_ms, duration_s).

Reads HF_TOKEN and (optionally) HF_REPO_ID from .env in the project root.

    .venv/bin/python -m pipeline.push_to_hub --dry-run --limit 5
        # builds a 5-row dataset locally and prints it - no network calls, safe to run anytime

    .venv/bin/python -m pipeline.push_to_hub --repo-id yourname/call-transcript-intent-data --limit 50
        # PUSHES a 50-row test slice to the real Hub (private by default) - use this to sanity
        # check the real thing on the Hub before pushing everything

    .venv/bin/python -m pipeline.push_to_hub --repo-id yourname/call-transcript-intent-data
        # pushes the full dataset (private by default; add --public to make it public)

    .venv/bin/python -m pipeline.push_to_hub --append --repo-id yourname/call-transcript-intent-data
        # appends output_dir's data.jsonl (default: OUTPUT_DIR_V2) onto the dataset
        # ALREADY on the Hub: downloads it, concatenates the new rows, de-dupes by
        # `id` (existing rows win), and pushes the combined result back. Safe to
        # re-run - already-appended rows are recognized and not duplicated.

This only ever reads data.jsonl - it does not touch skipped.jsonl or
pending_intent.jsonl, so only fully transcribed+classified rows are included.
"""

import argparse
import json
import os
import random

from datasets import Audio, Dataset, Features, Value, concatenate_datasets, load_dataset
from huggingface_hub import HfApi

from dataset_paths import resolve_chunk_path
from . import config

FEATURES = Features(
    {
        "id": Value("string"),
        "oid": Value("string"),
        "conversation_id": Value("string"),
        "recording_url": Value("string"),
        "chunk_index": Value("int32"),
        "start_ms": Value("float32"),
        "end_ms": Value("float32"),
        "duration_s": Value("float32"),
        "transcript": Value("string"),
        "intent": Value("string"),
        # Native sample rate is preserved (no forced resampling) - the source
        # chunks are 8kHz mono PCM16.
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
            chunk_path = rec.pop("chunk_path")
            audio_path = resolve_chunk_path(chunk_path, audio_dir)
            if not audio_path.exists():
                print(f"[push_to_hub] WARNING: missing audio file, skipping: {chunk_path}")
                continue
            rec["id"] = f"{rec['conversation_id']}_{rec['chunk_index']}"
            rec["audio"] = str(audio_path)
            yield rec
            n += 1
            if limit is not None and n >= limit:
                return


def build_dataset(data_path, limit=None, audio_dir=None):
    return Dataset.from_generator(lambda: _iter_records(data_path, limit=limit, audio_dir=audio_dir), features=FEATURES)


def _iter_records_filtered(data_path, allowed_chunk_paths, audio_dir=None):
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk_path = rec.pop("chunk_path")
            if chunk_path not in allowed_chunk_paths:
                continue
            audio_path = resolve_chunk_path(chunk_path, audio_dir)
            if not audio_path.exists():
                print(f"[push_to_hub] WARNING: missing audio file, skipping: {chunk_path}")
                continue
            rec["id"] = f"{rec['conversation_id']}_{rec['chunk_index']}"
            rec["audio"] = str(audio_path)
            yield rec


def build_dataset_filtered(data_path, allowed_chunk_paths, audio_dir=None):
    return Dataset.from_generator(
        lambda: _iter_records_filtered(data_path, allowed_chunk_paths, audio_dir=audio_dir), features=FEATURES
    )


def _split_chunk_paths(data_path, eval_size, val_size, seed=42):
    """Shuffle (fixed seed, reproducible) then carve off eval_size, then
    val_size, with everything else going to train. Returns (eval, val, train)
    sets of chunk_path."""
    paths = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(json.loads(line)["chunk_path"])

    rng = random.Random(seed)
    rng.shuffle(paths)

    total = len(paths)
    if eval_size + val_size > total:
        raise ValueError(
            f"eval_size ({eval_size}) + val_size ({val_size}) exceeds total rows ({total})"
        )

    eval_paths = set(paths[:eval_size])
    val_paths = set(paths[eval_size : eval_size + val_size])
    train_paths = set(paths[eval_size + val_size :])
    return eval_paths, val_paths, train_paths


def _load_metadata(output_dir):
    meta_path = os.path.join(output_dir, config.METADATA_FILENAME)
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _build_readme(meta, repo_id, num_rows):
    intents_list = meta.get("intents", [])
    lines = [
        "---",
        "language:",
        "- hi",
        "task_categories:",
        "- automatic-speech-recognition",
        "- audio-classification",
        f"pretty_name: {meta.get('dataset_name', repo_id)}",
        "---",
        "",
        f"# {meta.get('dataset_name', repo_id)}",
        "",
        meta.get("description", ""),
        "",
        f"- **Rows**: {num_rows}",
        f"- **Source dataset**: `{meta.get('source_dataset_dir', '')}`",
        f"- **Transcription backend**: `{meta.get('transcription_endpoint', meta.get('transcription_backend', ''))}` "
        f"({meta.get('transcription_language', '')})",
        f"- **Intent model**: `{meta.get('intent_model', '')}`",
        "",
        "## Fields",
        "",
        "| Field | Description |",
        "|---|---|",
        "| `id` | stable unique row id: `{conversation_id}_{chunk_index}` |",
        "| `audio` | audio clip for this turn (native sample rate, no forced resampling) |",
    ]
    for name, desc in meta.get("features", {}).items():
        if name == "chunk_path":
            continue  # replaced by `audio` above in the pushed dataset
        lines.append(f"| `{name}` | {desc} |")
    lines += [
        "",
        "## Intents",
        "",
        ", ".join(f"`{n}`" for n in intents_list) or "(see data)",
        "",
    ]
    return "\n".join(lines)


def _push_readme(output_dir, repo_id, num_rows, token):
    meta = _load_metadata(output_dir)
    readme = _build_readme(meta, repo_id, num_rows)
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    HfApi(token=token).upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )


def push(output_dir=config.OUTPUT_DIR, repo_id=None, private=True, split="train", limit=None, token=None, audio_dir=None):
    repo_id = repo_id or config.HF_REPO_ID
    if not repo_id:
        raise ValueError("no repo_id given - pass --repo-id or set HF_REPO_ID in .env")
    token = token or config.HF_TOKEN

    data_path = os.path.join(output_dir, config.DATA_FILENAME)
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    print(f"[push_to_hub] building dataset from {data_path}" + (f" (limit={limit})" if limit else ""))
    ds = build_dataset(data_path, limit=limit, audio_dir=audio_dir)
    print(f"[push_to_hub] {len(ds)} rows, features: {list(ds.features.keys())}")

    print(f"[push_to_hub] pushing to {repo_id} (private={private}, split={split})...")
    ds.push_to_hub(repo_id, private=private, split=split, token=token)

    _push_readme(output_dir, repo_id, len(ds), token)
    print(f"[push_to_hub] done -> https://huggingface.co/datasets/{repo_id}")
    return {"repo_id": repo_id, "num_rows": len(ds)}


def _dedupe_keep_first(ds, key="id"):
    """Drop rows whose `key` value already appeared earlier in the dataset.
    Row order matters: concatenate the existing (already-published) rows
    BEFORE the new ones so the existing copy wins on a collision."""
    seen = set()

    def _keep(key_value):
        if key_value in seen:
            return False
        seen.add(key_value)
        return True

    return ds.filter(_keep, input_columns=[key])


def append(output_dir=config.OUTPUT_DIR_V2, repo_id=None, split="train", limit=None, token=None, audio_dir=None):
    """Append output_dir's data.jsonl onto a dataset ALREADY pushed to the Hub.

    Downloads the current live dataset, concatenates the new rows, de-dupes by
    `id` (existing rows win on a collision), and pushes the combined result
    back to the same repo/split. Safe to re-run: rows already appended in a
    previous call are recognized via `id` and not duplicated.
    """
    repo_id = repo_id or config.HF_REPO_ID
    if not repo_id:
        raise ValueError("no repo_id given - pass --repo-id or set HF_REPO_ID in .env")
    token = token or config.HF_TOKEN

    data_path = os.path.join(output_dir, config.DATA_FILENAME)
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    print(f"[push_to_hub] loading existing dataset {repo_id} (split={split})...")
    existing_ds = load_dataset(repo_id, split=split, token=token)
    print(f"[push_to_hub] existing: {len(existing_ds)} rows")

    print(f"[push_to_hub] building new rows from {data_path}" + (f" (limit={limit})" if limit else ""))
    new_ds = build_dataset(data_path, limit=limit, audio_dir=audio_dir)
    print(f"[push_to_hub] new: {len(new_ds)} rows")

    combined = concatenate_datasets([existing_ds, new_ds])
    combined = _dedupe_keep_first(combined, key="id")
    added = len(combined) - len(existing_ds)
    print(f"[push_to_hub] combined: {len(combined)} rows ({added} net new after de-dup)")

    print(f"[push_to_hub] pushing combined dataset to {repo_id} (split={split})...")
    combined.push_to_hub(repo_id, split=split, token=token)

    _push_readme(output_dir, repo_id, len(combined), token)
    print(f"[push_to_hub] done -> https://huggingface.co/datasets/{repo_id}")
    return {
        "repo_id": repo_id,
        "existing_rows": len(existing_ds),
        "new_rows_added": added,
        "total_rows": len(combined),
    }


def delete_split(repo_id, split, token=None):
    """Delete a split's parquet shard(s) from an already-pushed dataset repo.
    This is a real deletion on the Hub, not locally reversible - only call it
    once you've confirmed the splits that should replace it are properly in
    place."""
    token = token or config.HF_TOKEN
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    targets = [f for f in files if f.startswith(f"data/{split}-") and f.endswith(".parquet")]
    if not targets:
        print(f"[push_to_hub] no files found for split '{split}' - nothing to delete")
        return []
    for f in targets:
        api.delete_file(path_in_repo=f, repo_id=repo_id, repo_type="dataset")
        print(f"[push_to_hub] deleted {f}")
    return targets


def split_and_push(
    output_dir=config.OUTPUT_DIR_V3,
    audio_dir=None,
    repo_id=None,
    eval_size=1000,
    val_size=10000,
    seed=42,
    delete_old_splits=("train", "v1", "v2"),
    private=True,
    token=None,
):
    """Carve output_dir's data.jsonl into eval/validation/train splits (fixed
    seed, so reproducible), push each to its own split on repo_id, then delete
    the given old splits - meant to replace v1/v2/the old merged "train" with
    ONLY this data, reorganized as a standard eval/validation/train layout.
    """
    repo_id = repo_id or config.HF_REPO_ID
    if not repo_id:
        raise ValueError("no repo_id given - pass --repo-id or set HF_REPO_ID in .env")
    token = token or config.HF_TOKEN

    data_path = os.path.join(output_dir, config.DATA_FILENAME)
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    eval_paths, val_paths, train_paths = _split_chunk_paths(data_path, eval_size, val_size, seed=seed)
    print(
        f"[push_to_hub] split sizes -> eval={len(eval_paths)} validation={len(val_paths)} "
        f"train={len(train_paths)}"
    )

    # Delete old splits FIRST, before pushing anything new. The new "train"
    # split (remainder) uses the same name as the old merged one - deleting
    # AFTER pushing would match and remove the brand-new files too, since
    # delete_split() can only key off the split name, not push recency.
    for split in delete_old_splits:
        delete_split(repo_id, split, token=token)

    counts = {}
    for split_name, allowed in (("eval", eval_paths), ("validation", val_paths), ("train", train_paths)):
        print(f"[push_to_hub] building '{split_name}' ({len(allowed)} rows)...")
        ds = build_dataset_filtered(data_path, allowed, audio_dir=audio_dir)
        print(f"[push_to_hub] pushing '{split_name}' ({len(ds)} rows) to {repo_id}...")
        ds.push_to_hub(repo_id, private=private, split=split_name, token=token)
        counts[split_name] = len(ds)

    _push_readme(output_dir, repo_id, sum(counts.values()), token)

    print(f"[push_to_hub] done -> https://huggingface.co/datasets/{repo_id}")
    return counts


def _default_split_for(output_dir):
    """Each generation pass gets its own split by default - v1/v2/v3 stay
    distinct and inspectable rather than getting silently merged into one
    "train" split, unless the caller explicitly asks for a different split."""
    return {
        config.OUTPUT_DIR: "v1",
        config.OUTPUT_DIR_V2: "v2",
        config.OUTPUT_DIR_V3: "v3",
    }.get(output_dir, "train")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"default: {config.OUTPUT_DIR_V2} with --append, else {config.OUTPUT_DIR}",
    )
    parser.add_argument(
        "--repo-id", default=None, help="e.g. yourname/call-transcript-intent-data (or set HF_REPO_ID in .env)"
    )
    parser.add_argument("--public", action="store_true", help="push as a public dataset (default: private)")
    parser.add_argument(
        "--append",
        action="store_true",
        help="append onto a dataset already pushed to the Hub instead of pushing fresh (see module docstring)",
    )
    parser.add_argument(
        "--split", default=None, help="default: v1/v2/v3 based on --output-dir, else 'train'"
    )
    parser.add_argument("--limit", type=int, default=None, help="only include the first N rows")
    parser.add_argument("--audio-dir", default=None, help="audio folder for relative chunk_path values; defaults to <output-dir>/audio")
    parser.add_argument(
        "--dry-run", action="store_true", help="build the dataset locally and print a sample - no network calls"
    )
    parser.add_argument(
        "--delete-split",
        metavar="SPLIT_NAME",
        default=None,
        help="delete this split's files from --repo-id and exit (real deletion on the Hub - use with care)",
    )
    parser.add_argument(
        "--split-final",
        action="store_true",
        help=(
            "carve --output-dir's data.jsonl (default: OUTPUT_DIR_V3) into eval/validation/train "
            "splits, push each, then delete --delete-old-splits (default: train,v1,v2) - "
            "use this to replace older splits with ONLY this data (see module docstring)"
        ),
    )
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--delete-old-splits",
        default="train,v1,v2",
        help="comma-separated split names to delete after --split-final pushes the new ones",
    )
    args = parser.parse_args()
    audio_dir = args.audio_dir or os.path.join(args.output_dir or (config.OUTPUT_DIR_V2 if args.append else config.OUTPUT_DIR), "audio")

    if args.delete_split:
        if not args.repo_id and not config.HF_REPO_ID:
            raise ValueError("no repo_id given - pass --repo-id or set HF_REPO_ID in .env")
        delete_split(args.repo_id or config.HF_REPO_ID, args.delete_split)
        return

    if args.split_final:
        result = split_and_push(
            output_dir=args.output_dir or config.OUTPUT_DIR_V3,
            repo_id=args.repo_id,
            eval_size=args.eval_size,
            val_size=args.val_size,
            seed=args.seed,
            delete_old_splits=[s.strip() for s in args.delete_old_splits.split(",") if s.strip()],
            private=not args.public,
            audio_dir=audio_dir,
        )
        print(f"[push_to_hub] result -> {result}")
        return

    output_dir = args.output_dir or (config.OUTPUT_DIR_V2 if args.append else config.OUTPUT_DIR)
    split = args.split or _default_split_for(output_dir)

    if args.dry_run:
        data_path = os.path.join(output_dir, config.DATA_FILENAME)
        ds = build_dataset(data_path, limit=args.limit or 5, audio_dir=audio_dir)
        print(ds)
        print("\nsample row 0:")
        print(ds[0])
        return

    if args.append:
        result = append(
            output_dir=output_dir,
            repo_id=args.repo_id,
            split=split,
            limit=args.limit,
            audio_dir=audio_dir,
        )
    else:
        result = push(
            output_dir=output_dir,
            repo_id=args.repo_id,
            private=not args.public,
            split=split,
            limit=args.limit,
            audio_dir=audio_dir,
        )
    print(f"[push_to_hub] result -> {result}")


if __name__ == "__main__":
    main()
