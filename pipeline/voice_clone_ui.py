"""Curation UI for voice-clone samples.

Walks through pipeline.voice_clone_candidates' output (one candidate per
conversation_id, already filtered to transcript >20 chars and duration >3s),
lets you listen, pick a start/end range to keep (default: first
config.VC_DEFAULT_CLIP_SECONDS), and Yes/No it.

Yes -> trims the audio to that range, saves it under VC_SAMPLES_DIR, and
appends a record to voice_clone_users.json.
No  -> marks it rejected (tracked in voice_clone_rejected.json) so it isn't
shown again.

Run with:
    .venv/bin/python -m pipeline.voice_clone_candidates build   # once, or whenever you want to refresh candidates
    .venv/bin/python -m pipeline.voice_clone_ui
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone

import gradio as gr
import soundfile as sf

from . import config


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load_json_list(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_json_list(path, items):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_candidates():
    if not os.path.exists(config.VC_CANDIDATES_PATH):
        raise FileNotFoundError(
            f"{config.VC_CANDIDATES_PATH} not found - run "
            "`python -m pipeline.voice_clone_candidates build` first"
        )
    with open(config.VC_CANDIDATES_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_reviewed_conversation_ids():
    accepted = {u["conversation_id"] for u in _load_json_list(config.VOICE_CLONE_USERS_JSON)}
    rejected = {r["conversation_id"] for r in _load_json_list(config.VC_REJECTED_PATH)}
    return accepted | rejected


def _next_unreviewed_index(candidates, reviewed, start_from):
    for i in range(start_from, len(candidates)):
        if candidates[i]["conversation_id"] not in reviewed:
            return i
    return None


def accept_candidate(candidate, start_s, end_s):
    os.makedirs(config.VC_SAMPLES_DIR, exist_ok=True)
    audio, sr = sf.read(candidate["chunk_path"], dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration_s = len(audio) / sr
    start_s = max(0.0, min(start_s, duration_s))
    end_s = max(start_s + 0.1, min(end_s, duration_s))
    clip = audio[int(start_s * sr) : int(end_s * sr)]

    sample_id = uuid.uuid4().hex[:12]
    sample_path = os.path.join(config.VC_SAMPLES_DIR, f"{candidate['conversation_id']}_{sample_id}.wav")
    sf.write(sample_path, clip, sr)

    users = _load_json_list(config.VOICE_CLONE_USERS_JSON)
    users.append(
        {
            "conversation_id": candidate["conversation_id"],
            "oid": candidate.get("oid"),
            "source_chunk_path": candidate["chunk_path"],
            "sample_path": sample_path,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "transcript": candidate.get("transcript"),
            "accepted_at": _now(),
        }
    )
    _save_json_list(config.VOICE_CLONE_USERS_JSON, users)
    return sample_path


def reject_candidate(candidate):
    rejected = _load_json_list(config.VC_REJECTED_PATH)
    rejected.append({"conversation_id": candidate["conversation_id"], "rejected_at": _now()})
    _save_json_list(config.VC_REJECTED_PATH, rejected)


def build_app():
    candidates = load_candidates()
    reviewed = load_reviewed_conversation_ids()

    with gr.Blocks(title="Voice-clone sample curation") as demo:
        gr.Markdown("## Voice-clone sample curation")

        state_index = gr.State(_next_unreviewed_index(candidates, reviewed, 0))
        state_reviewed = gr.State(reviewed)

        progress_md = gr.Markdown()
        with gr.Row():
            info_md = gr.Markdown()

        audio_player = gr.Audio(label="Full chunk", type="filepath")
        transcript_md = gr.Markdown()

        with gr.Row():
            start_s_input = gr.Number(label="Start (s)", value=0.0, precision=2)
            end_s_input = gr.Number(label="End (s)", value=config.VC_DEFAULT_CLIP_SECONDS, precision=2)
            preview_btn = gr.Button("Preview trimmed range")

        preview_player = gr.Audio(label="Trimmed preview", type="filepath")

        with gr.Row():
            no_btn = gr.Button("No - reject", variant="stop")
            yes_btn = gr.Button("Yes - use for voice clone", variant="primary")

        status_md = gr.Markdown()

        def _render(index):
            total = len(candidates)
            if index is None:
                return (
                    f"**Done - no more unreviewed candidates.** ({total} total, "
                    f"{len(_load_json_list(config.VOICE_CLONE_USERS_JSON))} accepted so far)",
                    "",
                    None,
                    "",
                    0.0,
                    config.VC_DEFAULT_CLIP_SECONDS,
                    None,
                    "",
                )
            c = candidates[index]
            duration = c.get("duration_s", 0)
            default_end = min(config.VC_DEFAULT_CLIP_SECONDS, duration)
            progress = f"**Candidate {index + 1} / {total}**"
            info = (
                f"conversation_id: `{c['conversation_id']}` &nbsp;|&nbsp; "
                f"duration: {duration:.2f}s &nbsp;|&nbsp; intent: `{c.get('intent', '')}`"
            )
            transcript = f"transcript (for context only): {c.get('transcript', '')}"
            return progress, info, c["chunk_path"], transcript, 0.0, default_end, None, ""

        demo.load(
            _render,
            inputs=[state_index],
            outputs=[progress_md, info_md, audio_player, transcript_md, start_s_input, end_s_input, preview_player, status_md],
        )

        def do_preview(index, start_s, end_s):
            if index is None:
                return None
            c = candidates[index]
            audio, sr = sf.read(c["chunk_path"], dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            duration_s = len(audio) / sr
            s = max(0.0, min(start_s, duration_s))
            e = max(s + 0.05, min(end_s, duration_s))
            clip = audio[int(s * sr) : int(e * sr)]
            tmp_path = os.path.join(config.DATA_DIR, "_vc_preview_tmp.wav")
            os.makedirs(config.DATA_DIR, exist_ok=True)
            sf.write(tmp_path, clip, sr)
            return tmp_path

        preview_btn.click(do_preview, inputs=[state_index, start_s_input, end_s_input], outputs=[preview_player])

        def do_yes(index, start_s, end_s, reviewed_set):
            if index is None:
                return (*_render(None), index, reviewed_set)
            c = candidates[index]
            sample_path = accept_candidate(c, start_s, end_s)
            reviewed_set = set(reviewed_set) | {c["conversation_id"]}
            next_index = _next_unreviewed_index(candidates, reviewed_set, index + 1)
            rendered = _render(next_index)
            status = f"accepted -> saved to `{sample_path}`"
            return (*rendered[:-1], status, next_index, reviewed_set)

        def do_no(index, reviewed_set):
            if index is None:
                return (*_render(None), index, reviewed_set)
            c = candidates[index]
            reject_candidate(c)
            reviewed_set = set(reviewed_set) | {c["conversation_id"]}
            next_index = _next_unreviewed_index(candidates, reviewed_set, index + 1)
            rendered = _render(next_index)
            status = f"rejected `{c['conversation_id']}`"
            return (*rendered[:-1], status, next_index, reviewed_set)

        yes_btn.click(
            do_yes,
            inputs=[state_index, start_s_input, end_s_input, state_reviewed],
            outputs=[
                progress_md, info_md, audio_player, transcript_md,
                start_s_input, end_s_input, preview_player, status_md,
                state_index, state_reviewed,
            ],
        )
        no_btn.click(
            do_no,
            inputs=[state_index, state_reviewed],
            outputs=[
                progress_md, info_md, audio_player, transcript_md,
                start_s_input, end_s_input, preview_player, status_md,
                state_index, state_reviewed,
            ],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_app()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
