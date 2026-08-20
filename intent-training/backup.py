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

import gradio as gr
import numpy as np
import torch
from librosa import resample as librosa_resample

import UI_CONFIG as cfg
from model import WhisperIntentClassification
from whisper.audio import N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim

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

    no_op = (gr.update(), gr.update())  # (result, history)

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
                    "duration_s": f"{duration_ms / 1000:.2f}",
                    "response_ms": f"{response_ms:.0f}",
                })
                state["history"] = state["history"][: cfg.HISTORY_LENGTH]
                history_update = gr.update(value=_history_table_html(state["history"]))

            state["speech_active"] = False
            state["buffer"] = []

    return state, _vad_html(state["speech_active"] or is_loud), result_update, history_update


def _history_table_html(history):
    """A single hand-rolled <table> instead of gr.Dataframe - the latter
    renders its header and body as separate DOM regions with independently
    computed column widths, which drift out of alignment with long variable-
    length text (like our intent names). One real table can't drift."""
    cols = ["time", "intent", "confidence", "duration_s", "response_ms"]
    header = "".join(f"<th>{c}</th>" for c in cols)
    if not history:
        body = f"<tr><td colspan='{len(cols)}' style='text-align:center;color:#6b7280;'>No classifications yet.</td></tr>"
    else:
        body = "".join(
            "<tr>"
            f"<td>{h['time']}</td><td>{h['intent']}</td><td>{h['confidence']}</td>"
            f"<td>{h['duration_s']}</td><td>{h['response_ms']}</td>"
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
            "<div style='padding:14px;border-radius:10px;background:#0f3d24;"
            "border:1px solid #22c55e;color:#4ade80;font-weight:600;"
            "font-size:1.1rem;text-align:center;'>&#128308; SPEECH DETECTED</div>"
        )
    return (
        "<div style='padding:14px;border-radius:10px;background:#1e1e1e;"
        "border:1px solid #3a3a3a;color:#9ca3af;font-weight:600;"
        "font-size:1.1rem;text-align:center;'>&#9899; listening...</div>"
    )


# Only hide the footer and style our own hand-rolled history table - let
# Gradio's own theme (toggled via the ?__theme=dark URL param) handle every
# other color/shadow. Forcing raw colors on Gradio's own components with
# !important fights the theme's computed shadow/border colors and causes a
# muddy "double-darkened" look plus repaint jank on updates.
DARK_CSS = """
footer { display: none !important; }
.history-table { width: 100%; border-collapse: collapse; }
.history-table th, .history-table td {
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #333;
}
.history-table th { font-weight: 600; color: #9ca3af; }
"""

with gr.Blocks(title="Intent Classifier") as demo:
    gr.Markdown("##Call Intent Classifier")
    gr.Markdown(
        f"Model: `{os.path.basename(cfg.MODEL_PATH)}` on **{DEVICE}** · "
        f"{N_CLASS} intents"
        # f"{N_CLASS} intents · energy-based VAD (threshold {cfg.VAD_DB_THRESHOLD} dBFS)"
    )

    state = gr.State(_new_state())

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["microphone"], streaming=True, type="numpy", label="Microphone"
            )
            vad_indicator = gr.HTML(_vad_html(False))
        with gr.Column(scale=1):
            label_out = gr.Label(num_top_classes=5, label="Predicted intent")

    gr.Markdown(f"### Last {cfg.HISTORY_LENGTH} intents")
    history_out = gr.HTML(_history_table_html([]))

    audio_in.stream(
        fn=on_audio_chunk,
        inputs=[audio_in, state],
        outputs=[state, vad_indicator, label_out, history_out],
        stream_every=cfg.STREAM_EVERY_S,
        # serialize calls for this event - otherwise an overlapping call
        # (classification can take 50-250ms+) can read a stale `state`
        # snapshot and clobber the previous call's history update, silently
        # dropping detections from the table.
        concurrency_limit=1,
    )

# Force Gradio's own dark theme (its native ?__theme=dark mechanism) instead
# of overriding colors ourselves - redirects once, before the app renders, so
# there's no flash-of-light-mode.
FORCE_DARK_HEAD = """
<script>
if (!location.search.includes('__theme')) {
  const sep = location.search ? '&' : '?';
  location.replace(location.pathname + location.search + sep + '__theme=dark');
}
</script>
"""

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=cfg.SERVER_PORT,
        # share=os.environ.get("GRADIO_SHARE") == "1",
        theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate"),
        css=DARK_CSS,
        head=FORCE_DARK_HEAD,
        share=True
    )
