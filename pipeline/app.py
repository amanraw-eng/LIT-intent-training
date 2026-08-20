"""Gradio UI to browse the generated call_trascript_intent_data dataset.

Run with:
    .venv/bin/python -m pipeline.app

Works while generation is still running in another process - hit Refresh to
reload the latest rows from disk.
"""

import argparse
import json
import os

import gradio as gr
import pandas as pd

from . import config
from .intents import INTENT_NAMES

DIR_CHOICES = {
    "Main (call_trascript_intent_data)": config.OUTPUT_DIR,
    "Sample (call_trascript_intent_data_sample)": config.SAMPLE_OUTPUT_DIR,
}

DISPLAY_COLS = [
    "chunk_index",
    "conversation_id",
    "start_ms",
    "end_ms",
    "duration_s",
    "intent",
    "transcript",
]

PAGE_SIZE_CHOICES = [25, 50, 100, 200]


def load_records(output_dir):
    data_path = os.path.join(output_dir, config.DATA_FILENAME)
    records = []
    if os.path.exists(data_path):
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return records


def load_metadata(output_dir):
    meta_path = os.path.join(output_dir, config.METADATA_FILENAME)
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def stats_markdown(output_dir, records):
    meta = load_metadata(output_dir)
    lines = [f"**Loaded {len(records)} rows** from `{output_dir}`"]
    if meta:
        lines.append(
            f"status: `{meta.get('status', 'unknown')}` · "
            f"processed: {meta.get('num_chunks_processed', '?')}/{meta.get('num_chunks_total', '?')} "
            f"chunks across {meta.get('num_conversations_total', '?')} conversations · "
            f"last updated: {meta.get('last_updated', '?')}"
        )
        if meta.get("last_error"):
            lines.append(f"last error: `{meta['last_error']}`")
    if records:
        counts = pd.Series([r.get("intent") for r in records]).value_counts()
        top = ", ".join(f"{k}: {v}" for k, v in counts.head(8).items())
        lines.append(f"top intents -> {top}")
    return "\n\n".join(lines)


def filter_records(records, intent_filter, search_text):
    out = records
    if intent_filter:
        wanted = set(intent_filter)
        out = [r for r in out if r.get("intent") in wanted]
    if search_text:
        s = search_text.lower()
        out = [r for r in out if s in (r.get("transcript") or "").lower()]
    return out


def page_slice(records, page, page_size):
    n_pages = max(1, (len(records) + page_size - 1) // page_size)
    page = max(0, min(page, n_pages - 1))
    start = page * page_size
    return records[start : start + page_size], page, n_pages


def to_dataframe(page_records):
    if not page_records:
        return pd.DataFrame(columns=DISPLAY_COLS)
    df = pd.DataFrame(page_records)
    for col in DISPLAY_COLS:
        if col not in df.columns:
            df[col] = None
    return df[DISPLAY_COLS]


def build_app():
    with gr.Blocks(title="call_trascript_intent_data viewer") as demo:
        gr.Markdown("## Call transcript + intent dataset viewer")

        all_records = gr.State([])
        filtered_records = gr.State([])
        page_records_state = gr.State([])
        page_state = gr.State(0)

        with gr.Row():
            dir_dropdown = gr.Dropdown(
                choices=list(DIR_CHOICES.keys()),
                value=list(DIR_CHOICES.keys())[0],
                label="Dataset",
            )
            refresh_btn = gr.Button("Refresh")

        stats_md = gr.Markdown()

        with gr.Row():
            intent_filter = gr.Dropdown(
                choices=INTENT_NAMES, multiselect=True, label="Filter by intent"
            )
            search_box = gr.Textbox(label="Search transcript contains...")
            page_size_dd = gr.Dropdown(
                choices=PAGE_SIZE_CHOICES, value=PAGE_SIZE_CHOICES[1], label="Page size"
            )

        with gr.Row():
            prev_btn = gr.Button("< Prev")
            page_label = gr.Markdown("page 1 / 1")
            next_btn = gr.Button("Next >")

        table = gr.Dataframe(
            value=to_dataframe([]),
            interactive=False,
            wrap=True,
        )

        gr.Markdown("### Selected row")
        with gr.Row():
            audio_player = gr.Audio(label="Chunk audio", type="filepath")
            with gr.Column():
                detail_md = gr.Markdown()

        def do_load(dir_key):
            output_dir = DIR_CHOICES[dir_key]
            records = load_records(output_dir)
            return records, stats_markdown(output_dir, records)

        def do_filter(records, intents_sel, search_text, page_size):
            filtered = filter_records(records, intents_sel, search_text)
            page_recs, page, n_pages = page_slice(filtered, 0, page_size)
            return (
                filtered,
                page_recs,
                0,
                to_dataframe(page_recs),
                f"page {page + 1} / {n_pages}",
            )

        def do_page(filtered, page, page_size, delta):
            page_recs, page, n_pages = page_slice(filtered, page + delta, page_size)
            return page_recs, page, to_dataframe(page_recs), f"page {page + 1} / {n_pages}"

        def do_select(evt: gr.SelectData, page_recs):
            if evt is None or evt.index is None:
                return None, ""
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if row_idx is None or row_idx >= len(page_recs):
                return None, ""
            rec = page_recs[row_idx]
            detail = "\n\n".join(
                [
                    f"**oid**: {rec.get('oid')}",
                    f"**conversation_id**: {rec.get('conversation_id')}",
                    f"**chunk_path**: `{rec.get('chunk_path')}`",
                    f"**start_ms / end_ms**: {rec.get('start_ms')} / {rec.get('end_ms')}",
                    f"**duration_s**: {rec.get('duration_s')}",
                    f"**intent**: `{rec.get('intent')}`",
                    f"**transcript**: {rec.get('transcript')}",
                ]
            )
            audio_path = rec.get("chunk_path")
            if not audio_path or not os.path.exists(audio_path):
                audio_path = None
            return audio_path, detail

        refresh_and_filter_inputs = [dir_dropdown]

        def refresh_all(dir_key, intents_sel, search_text, page_size):
            records, stats = do_load(dir_key)
            filtered, page_recs, page, df, page_label_text = do_filter(
                records, intents_sel, search_text, page_size
            )
            return records, stats, filtered, page_recs, page, df, page_label_text

        dir_dropdown.change(
            refresh_all,
            inputs=[dir_dropdown, intent_filter, search_box, page_size_dd],
            outputs=[
                all_records,
                stats_md,
                filtered_records,
                page_records_state,
                page_state,
                table,
                page_label,
            ],
        )
        refresh_btn.click(
            refresh_all,
            inputs=[dir_dropdown, intent_filter, search_box, page_size_dd],
            outputs=[
                all_records,
                stats_md,
                filtered_records,
                page_records_state,
                page_state,
                table,
                page_label,
            ],
        )

        for trigger in (intent_filter.change, search_box.submit, page_size_dd.change):
            trigger(
                do_filter,
                inputs=[all_records, intent_filter, search_box, page_size_dd],
                outputs=[filtered_records, page_records_state, page_state, table, page_label],
            )

        prev_btn.click(
            lambda filtered, page, page_size: do_page(filtered, page, page_size, -1),
            inputs=[filtered_records, page_state, page_size_dd],
            outputs=[page_records_state, page_state, table, page_label],
        )
        next_btn.click(
            lambda filtered, page, page_size: do_page(filtered, page, page_size, 1),
            inputs=[filtered_records, page_state, page_size_dd],
            outputs=[page_records_state, page_state, table, page_label],
        )

        table.select(
            do_select,
            inputs=[page_records_state],
            outputs=[audio_player, detail_md],
        )

        demo.load(
            refresh_all,
            inputs=[dir_dropdown, intent_filter, search_box, page_size_dd],
            outputs=[
                all_records,
                stats_md,
                filtered_records,
                page_records_state,
                page_state,
                table,
                page_label,
            ],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_app()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
