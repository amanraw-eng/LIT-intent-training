"""Live mic demo: VAD-gated intent classification.

Streams microphone audio from the browser, uses a simple energy-based VAD to
find utterance boundaries, and classifies each completed utterance with the
trained WhisperIntentClassification model.

Run:
    .venv/bin/python intent-training/app_ui.py
"""

import json
import os
import time
import urllib.parse

import gradio as gr
import numpy as np
import torch
from librosa import resample as librosa_resample

import UI_CONFIG as cfg
from model import WhisperIntentClassification
from whisper.audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram, pad_or_trim

DEVICE = cfg.DEVICE if torch.cuda.is_available() else "cpu"

with open(cfg.INTENT_MAP_PATH, "r", encoding="utf-8") as f:
    INTENT_TO_IDX = json.load(f)
IDX_TO_INTENT = {v: k for k, v in INTENT_TO_IDX.items()}
N_CLASS = len(INTENT_TO_IDX)


def load_model():
    model = WhisperIntentClassification(cfg.MODEL_TYPE, n_class=N_CLASS)
    checkpoint = torch.load(cfg.MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = {
        key[len("model."):]: value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


print(f"[app_ui] loading {cfg.MODEL_PATH} on {DEVICE} ({N_CLASS} intents)")
MODEL = load_model()
print("[app_ui] model ready")


@torch.no_grad()
def classify_utterance(audio_16k):
    audio = pad_or_trim(audio_16k.astype(np.float32), N_SAMPLES)
    mel = log_mel_spectrogram(audio).to(DEVICE)
    logits = MODEL(mel.unsqueeze(0))
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return {IDX_TO_INTENT[i]: float(probs[i]) for i in range(N_CLASS)}


# ---------------------------------------------------------------------------
# audio folder review: browse a static folder, predict, correct, save feedback
# ---------------------------------------------------------------------------

def list_audio_files():
    """Never raises - a missing/unset/empty folder just yields []."""
    folder = cfg.AUDIO_FOLDER_PATH
    if not folder:
        return []
    try:
        if not os.path.isdir(folder):
            return []
        return sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(cfg.AUDIO_EXTENSIONS)
        )
    except OSError:
        return []


def _audio_file_path(filename):
    if not filename or not cfg.AUDIO_FOLDER_PATH:
        return None
    path = os.path.join(cfg.AUDIO_FOLDER_PATH, filename)
    return os.path.abspath(path) if os.path.exists(path) else None


@torch.no_grad()
def classify_file(filename):
    path = _audio_file_path(filename)
    if path is None:
        return None
    audio = load_audio(path, sr=cfg.TARGET_SAMPLE_RATE)
    return classify_utterance(audio)


def save_feedback(filename, predicted_intent, corrected_intent):
    if not filename:
        return "⚠️ No file."
    if not corrected_intent:
        return "⚠️ Select intent."

    record = {
        "id": filename,
        "audio_path": _audio_file_path(filename) or filename,
        "predicted_intent": predicted_intent,
        "corrected_intent": corrected_intent,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        os.makedirs(os.path.dirname(cfg.FEEDBACK_LOG_PATH), exist_ok=True)
        with open(cfg.FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        return f"❌ Error: {e}"
    return f"✅ Saved -> **{corrected_intent}**"


def _to_mono_float32(data):
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    return data


def _dbfs(samples):
    rms = np.sqrt(np.mean(np.square(samples)) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def _new_state():
    return {
        "speech_active": False,
        "last_speech_ts": None,
        "utterance_start_ts": None,
        "buffer": [],
        "history": [],
    }


def on_audio_chunk(chunk, state):
    if state is None:
        state = _new_state()

    no_op = (gr.update(), gr.update())

    if chunk is None:
        result_update, history_update = no_op
        return state, _vad_html(False), result_update, history_update

    sr, data = chunk
    samples = _to_mono_float32(np.asarray(data))
    if sr != cfg.TARGET_SAMPLE_RATE:
        samples = librosa_resample(samples, orig_sr=sr, target_sr=cfg.TARGET_SAMPLE_RATE)

    now = time.time()
    is_loud = _dbfs(samples) > cfg.VAD_DB_THRESHOLD

    if is_loud:
        state["last_speech_ts"] = now
        if not state["speech_active"]:
            state["speech_active"] = True
            state["utterance_start_ts"] = now
            state["buffer"] = []

    result_update, history_update = no_op

    if state["speech_active"]:
        state["buffer"].append(samples)

        silence_ms = (now - state["last_speech_ts"]) * 1000
        utterance_ms = (now - state["utterance_start_ts"]) * 1000
        should_finalize = silence_ms >= cfg.SILENCE_TO_TRIGGER_MS or utterance_ms >= cfg.MAX_UTTERANCE_MS

        if should_finalize:
            utterance_audio = np.concatenate(state["buffer"]) if state["buffer"] else np.array([], dtype=np.float32)
            duration_ms = len(utterance_audio) / cfg.TARGET_SAMPLE_RATE * 1000

            if duration_ms >= cfg.MIN_UTTERANCE_MS:
                infer_start = time.perf_counter()
                probs = classify_utterance(utterance_audio)
                response_ms = (time.perf_counter() - infer_start) * 1000
                top_intent = max(probs, key=probs.get)
                result_update = gr.update(value=probs)

                state["history"].insert(0, {
                    "time": time.strftime("%H:%M:%S"),
                    "intent": top_intent,
                    "confidence": f"{probs[top_intent]:.2f}",
                    "duration_s": f"{duration_ms / 1000:.2f}s",
                    "response_ms": f"{response_ms:.0f}ms",
                })
                state["history"] = state["history"][: cfg.HISTORY_LENGTH]
                history_update = gr.update(value=_history_table_html(state["history"]))

            state["speech_active"] = False
            state["buffer"] = []

    return state, _vad_html(state["speech_active"] or is_loud), result_update, history_update


def _history_table_html(history):
    cols = ["Time", "Predicted Intent", "Confidence", "Duration", "Latency"]
    header = "".join(f"<th>{c}</th>" for c in cols)
    if not history:
        body = f"<tr><td colspan='{len(cols)}' style='text-align:center;color:#64748b;padding:16px;'>No classifications recorded yet. Speak into the microphone above.</td></tr>"
    else:
        body = "".join(
            "<tr>"
            f"<td><code>{h['time']}</code></td>"
            f"<td><strong style='color:#38bdf8;'>{h['intent']}</strong></td>"
            f"<td><span class='badge'>{h['confidence']}</span></td>"
            f"<td>{h['duration_s']}</td>"
            f"<td><code>{h['response_ms']}</code></td>"
            "</tr>"
            for h in history
        )
    return (
        "<table class='history-table'><thead><tr>" + header + "</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _vad_html(active):
    if active:
        return (
            "<div class='vad-status active'>"
            "<span class='status-dot active'></span>"
            "<span>SPEECH DETECTED</span>"
            "</div>"
        )
    return (
        "<div class='vad-status idle'>"
        "<span class='status-dot idle'></span>"
        "<span>Listening...</span>"
        "</div>"
    )


DARK_CSS = """
footer { display: none !important; }
.space-logo, .spaces-logo { display: none !important; }

/* VAD Indicator Box */
.vad-status {
    padding: 10px 14px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    transition: all 0.2s ease;
    margin-top: 8px;
}
.vad-status.active {
    background: #064e3b;
    border: 1px solid #10b981;
    color: #34d399;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
}
.vad-status.idle {
    background: #1e293b;
    border: 1px solid #334155;
    color: #94a3b8;
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}
.status-dot.active { background: #34d399; box-shadow: 0 0 8px #34d399; }
.status-dot.idle { background: #64748b; }

/* History Table Formatting */
.history-table { 
    width: 100%; 
    border-collapse: collapse; 
    margin-top: 8px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #334155;
}
.history-table th, .history-table td {
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #334155;
    font-size: 0.85rem;
}
.history-table th { 
    font-weight: 600; 
    color: #94a3b8; 
    background-color: #0f172a; 
}
.history-table tr:last-child td { border-bottom: none; }
.badge {
    background: #0284c7;
    color: #ffffff;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Row Container Layout - Edge to Edge, No Dead Space */
.audio-review-row {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    padding: 6px 12px !important;
    margin-bottom: 6px !important;
    gap: 8px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    overflow: visible !important;
}

.audio-review-row:hover, .audio-review-row:focus-within {
    z-index: 50 !important;
    border-color: #475569 !important;
}

/* Kill Gradio's default min-widths and internal container margins */
.audio-review-row > div,
.audio-review-row .block,
.audio-review-row .form {
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    min-width: 0 !important;
    flex-shrink: 1 !important;
}

/* Filename Fix: Single line, truncated with ... */
.file-name-cell {
    font-size: 0.78rem !important;
    line-height: 28px !important;
    height: 28px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    color: #cbd5e1 !important;
    margin: 0 !important;
    display: block !important;
}

/* Audio Player Container & Dark Theme Invert */
.audio-line-container {
    display: flex !important;
    align-items: center !important;
    height: 28px !important;
    width: 100% !important;
}
.audio-line-container audio {
    height: 28px !important;
    width: 100% !important;
    border-radius: 14px !important;
    filter: invert(0.88) hue-rotate(180deg) !important; /* Dark mode styling */
}

/* Compact Buttons */
.btn-compact button {
    height: 28px !important;
    min-height: 28px !important;
    max-height: 28px !important;
    line-height: 28px !important;
    padding: 0 12px !important;
    margin: 0 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    min-width: unset !important;
    width: 100% !important;
    border-radius: 4px !important;
}

.result-cell, .status-cell {
    font-size: 0.78rem !important;
    line-height: 28px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin: 0 !important;
}
.result-cell { color: #38bdf8 !important; }

/* Dropdown Compact Sizing & Drop Arrow Clearance */
.compact-dropdown {
    margin: 0 !important;
}
.compact-dropdown .wrap,
.compact-dropdown input,
.compact-dropdown .single-select {
    min-height: 28px !important;
    height: 28px !important;
    border-radius: 4px !important;
    font-size: 0.75rem !important;
}
.compact-dropdown input {
    padding-left: 8px !important;
    padding-right: 28px !important; /* Prevents text colliding with drop arrow */
    line-height: 28px !important;
}
.compact-dropdown .options {
    z-index: 9999 !important;
    background: #0f172a !important;
    border: 1px solid #3b82f6 !important;
    max-height: 180px !important;
}

.pagination-row {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 16px !important;
    margin-top: 14px !important;
}
.page-info { color: #94a3b8; font-size: 0.85rem; font-weight: 500; }
"""

with gr.Blocks(title="Intent Classifier", theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate"), css=DARK_CSS) as demo:
    gr.Markdown("## 🎙️ Call Intent Classifier")
    gr.Markdown(
        f"**Model:** `{os.path.basename(cfg.MODEL_PATH)}` · **Device:** `{DEVICE}` · "
        f"**Classes:** {N_CLASS} intents"
    )

    state = gr.State(_new_state())

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["microphone"], streaming=True, type="numpy", label="Live Microphone Input"
            )
            vad_indicator = gr.HTML(_vad_html(False))
        with gr.Column(scale=1):
            label_out = gr.Label(num_top_classes=5, label="Predicted Intent")

    gr.Markdown(f"### 📋 Recent History (Last {cfg.HISTORY_LENGTH})")
    history_out = gr.HTML(_history_table_html([]))

    audio_in.stream(
        fn=on_audio_chunk,
        inputs=[audio_in, state],
        outputs=[state, vad_indicator, label_out, history_out],
        stream_every=cfg.STREAM_EVERY_S,
        concurrency_limit=1,
    )

    gr.Markdown("---")
    gr.Markdown("## 📁 Audio Folder Review")

    page_state = gr.State(0)
    refresh_trigger = gr.State(0)

    @gr.render(inputs=[page_state, refresh_trigger])
    def render_audio_rows(page, _refresh):
        files = list_audio_files()

        if not files:
            if not cfg.AUDIO_FOLDER_PATH:
                gr.Markdown("_No `AUDIO_FOLDER_PATH` configured in `UI_CONFIG.py`._")
            else:
                gr.Markdown(f"_No audio files found in `{cfg.AUDIO_FOLDER_PATH}`. Add files and click refresh._")
            return

        page_size = cfg.AUDIO_LIST_PAGE_SIZE
        total_pages = max(1, (len(files) + page_size - 1) // page_size)
        page = max(0, min(page, total_pages - 1))
        chunk = files[page * page_size: (page + 1) * page_size]

        for filename in chunk:
            file_path = _audio_file_path(filename)
            if file_path:
                clean_path = urllib.parse.quote(file_path.replace("\\", "/"), safe="/:")
                audio_player_html = (
                    f'<div class="audio-line-container">'
                    f'<audio controls controlsList="nodownload" src="/file={clean_path}"></audio>'
                    f'</div>'
                )
            else:
                audio_player_html = '<span style="color:#ef4444;font-size:0.75rem;">Missing</span>'

            # Added min_width=0 to all columns to eliminate right-side dead space
            with gr.Row(elem_classes=["audio-review-row"]):
                gr.Markdown(f"`{filename}`", container=False, elem_classes=["file-name-cell"], scale=3, min_width=0)
                gr.HTML(audio_player_html, container=False, scale=3, min_width=0)
                predict_btn = gr.Button("Predict", size="sm", variant="secondary", elem_classes=["btn-compact"], scale=1, min_width=0)
                result_md = gr.Markdown("_not predicted_", container=False, elem_classes=["result-cell"], scale=2, min_width=0)
                correct_dd = gr.Dropdown(
                    choices=list(IDX_TO_INTENT.values()),
                    label=None,
                    show_label=False,
                    interactive=True,
                    container=False,
                    elem_classes=["compact-dropdown"],
                    scale=3,
                    min_width=0,
                )
                save_btn = gr.Button("Save", size="sm", variant="primary", elem_classes=["btn-compact"], scale=1, min_width=0)
                save_status = gr.Markdown("", container=False, elem_classes=["status-cell"], scale=1, min_width=0)

            predicted_state = gr.State(None)

            def _predict(fname=filename):
                probs = classify_file(fname)
                if probs is None:
                    return "_file missing_", None, None
                top = max(probs, key=probs.get)
                return f"**{top}** ({probs[top]:.2f})", top, top

            predict_btn.click(
                fn=_predict, 
                outputs=[result_md, correct_dd, predicted_state]
            )

            def _save(corrected, predicted, fname=filename):
                return save_feedback(fname, predicted, corrected)

            save_btn.click(
                fn=_save, 
                inputs=[correct_dd, predicted_state], 
                outputs=[save_status]
            )

        with gr.Row(elem_classes=["pagination-row"]):
            prev_btn = gr.Button("‹ Prev", size="sm", scale=0, min_width=0)
            gr.Markdown(
                f"Page **{page + 1}** of **{total_pages}** ({len(files)} files total)",
                elem_classes=["page-info"],
                container=False,
                min_width=0,
            )
            next_btn = gr.Button("Next ›", size="sm", scale=0, min_width=0)
            refresh_btn = gr.Button("🔄 Refresh", size="sm", scale=0, min_width=0)

        prev_btn.click(fn=lambda p: max(0, p - 1), inputs=[page_state], outputs=[page_state])
        next_btn.click(fn=lambda p: p + 1, inputs=[page_state], outputs=[page_state])
        refresh_btn.click(fn=lambda t: t + 1, inputs=[refresh_trigger], outputs=[refresh_trigger])


FORCE_DARK_HEAD = """
<script>
if (!location.search.includes('__theme')) {
  const sep = location.search ? '&' : '?';
  location.replace(location.pathname + location.search + sep + '__theme=dark');
}
</script>
"""

if __name__ == "__main__":
    allowed_paths = []
    if cfg.AUDIO_FOLDER_PATH and os.path.isdir(cfg.AUDIO_FOLDER_PATH):
        allowed_paths.append(os.path.abspath(cfg.AUDIO_FOLDER_PATH))

    demo.launch(
        server_name="0.0.0.0",
        server_port=cfg.SERVER_PORT,
        head=FORCE_DARK_HEAD,
        share=True,
        allowed_paths=allowed_paths,
    )
    