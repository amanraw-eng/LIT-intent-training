"""Find voice-clone sample candidates: real chunks from v3 with a long enough
transcript and duration - one candidate per conversation_id (the longest
qualifying chunk for that call), for the curation UI (pipeline/voice_clone_ui.py)
to review one representative sample per "user".

    .venv/bin/python -m pipeline.voice_clone_candidates build
"""

import argparse
import json
import os

from . import config


def find_candidates(source_data_path=None, min_chars=None, min_duration_s=None):
    source_data_path = source_data_path or os.path.join(config.OUTPUT_DIR_V3, config.DATA_FILENAME)
    min_chars = config.VC_MIN_TRANSCRIPT_CHARS if min_chars is None else min_chars
    min_duration_s = config.VC_MIN_DURATION_S if min_duration_s is None else min_duration_s

    best_by_conversation = {}
    with open(source_data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            transcript = (rec.get("transcript") or "").strip()
            duration_s = rec.get("duration_s") or 0
            if len(transcript) <= min_chars or duration_s <= min_duration_s:
                continue
            if not os.path.exists(rec.get("chunk_path", "")):
                continue
            conv_id = rec["conversation_id"]
            existing = best_by_conversation.get(conv_id)
            if existing is None or duration_s > existing["duration_s"]:
                best_by_conversation[conv_id] = rec

    return sorted(best_by_conversation.values(), key=lambda r: -r["duration_s"])


def build_and_save(output_path=None, **kwargs):
    output_path = output_path or config.VC_CANDIDATES_PATH
    candidates = find_candidates(**kwargs)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    print(f"[voice_clone_candidates] {len(candidates)} candidates -> {output_path}")
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build_p = sub.add_parser("build", help="Scan the source dataset and write the candidate list")
    build_p.add_argument("--source-data-path", default=None)
    build_p.add_argument("--min-chars", type=int, default=None)
    build_p.add_argument("--min-duration-s", type=float, default=None)
    build_p.add_argument("--output-path", default=None)
    args = parser.parse_args()
    build_and_save(
        output_path=args.output_path,
        source_data_path=args.source_data_path,
        min_chars=args.min_chars,
        min_duration_s=args.min_duration_s,
    )


if __name__ == "__main__":
    main()
