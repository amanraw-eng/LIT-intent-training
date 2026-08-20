"""Bulk-generate voice-clone reference samples from ALL candidates (no manual
review): trims each candidate to its first config.VC_DEFAULT_CLIP_SECONDS (or
the whole clip if shorter), saves to VC_SAMPLES_DIR, and writes
voice_clone_users.json as a flat list of the resulting audio paths - meant to
be randomly sampled from by augment_underrepresented.py for voice cloning.

    .venv/bin/python -m pipeline.voice_clone_candidates build   # if not already done
    .venv/bin/python -m pipeline.voice_clone_bulk run
"""

import argparse
import json
import os

import soundfile as sf

from . import config


def run(candidates_path=None, output_dir=None, users_json_path=None, clip_seconds=None, limit=None):
    candidates_path = candidates_path or config.VC_CANDIDATES_PATH
    output_dir = output_dir or config.VC_SAMPLES_DIR
    users_json_path = users_json_path or config.VOICE_CLONE_USERS_JSON
    clip_seconds = config.VC_DEFAULT_CLIP_SECONDS if clip_seconds is None else clip_seconds

    with open(candidates_path, encoding="utf-8") as f:
        candidates = json.load(f)
    if limit is not None:
        candidates = candidates[:limit]

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    n_failed = 0
    for c in candidates:
        try:
            audio, sr = sf.read(c["chunk_path"], dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            end_s = min(clip_seconds, len(audio) / sr)
            clip = audio[: int(end_s * sr)]
            out_path = os.path.join(output_dir, f"{c['conversation_id']}.wav")
            sf.write(out_path, clip, sr)
            paths.append(out_path)
        except Exception as e:
            print(f"[voice_clone_bulk] FAILED {c['conversation_id']}: {e}")
            n_failed += 1

    with open(users_json_path, "w", encoding="utf-8") as f:
        json.dump(paths, f, ensure_ascii=False, indent=2)

    print(f"[voice_clone_bulk] wrote {len(paths)} audio clips to {output_dir} ({n_failed} failed)")
    print(f"[voice_clone_bulk] wrote {len(paths)} paths -> {users_json_path}")
    return {"ok": len(paths), "failed": n_failed}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Generate clips + voice_clone_users.json for every candidate")
    run_p.add_argument("--candidates-path", default=None)
    run_p.add_argument("--output-dir", default=None)
    run_p.add_argument("--users-json-path", default=None)
    run_p.add_argument("--clip-seconds", type=float, default=None)
    run_p.add_argument("--limit", type=int, default=None, help="only process the first N candidates")
    args = parser.parse_args()
    run(
        candidates_path=args.candidates_path,
        output_dir=args.output_dir,
        users_json_path=args.users_json_path,
        clip_seconds=args.clip_seconds,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
