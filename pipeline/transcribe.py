import asyncio
import base64
import json
import re
import time

import websockets

from . import config
from .audio_utils import load_pcm16


class TranscriptionError(Exception):
    pass


class TranscriptionTimeout(TranscriptionError):
    """Raised when the server accepts config/audio/flush but never replies for this
    specific chunk. Observed on some short/quiet chunks - a per-chunk server quirk,
    not an outage, so callers should skip and continue rather than stop the run."""

    pass


def _split_frames(pcm_bytes, frame_size):
    frames = []
    for start in range(0, len(pcm_bytes), frame_size):
        piece = pcm_bytes[start : start + frame_size]
        if len(piece) < frame_size:
            piece = piece + b"\x00" * (frame_size - len(piece))
        frames.append(piece)
    return frames


def _clamp_chunk_bounds(duration_s):
    # BufferConfigOverride allows min in [0.5, 5.0], max in [2.0, 30.0].
    min_dur = min(max(duration_s, 0.5), 5.0)
    max_dur = min(max(duration_s + 0.5, 2.0), 30.0)
    if max_dur <= min_dur:
        max_dur = min_dur + 0.5
    return min_dur, max_dur


async def detect_frame_size_bytes():
    """Probe the server with a bad-size frame and parse the expected size from the error."""
    call_id = "frame-size-probe"
    uri = f"{config.LIT_WS_BASE}/{call_id}?language={config.LANGUAGE}"
    try:
        async with websockets.connect(uri, open_timeout=config.CONNECT_TIMEOUT_S) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "audio_frame",
                        "call_id": call_id,
                        "sequence": 0,
                        "timestamp": int(time.time() * 1000),
                        "audio": base64.b64encode(b"\x00\x00").decode(),
                        "sample_rate": config.TARGET_SAMPLE_RATE,
                    }
                )
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=config.RECV_TIMEOUT_S)
    except Exception as e:
        raise TranscriptionError(f"frame size probe failed: {e}") from e

    data = json.loads(raw)
    match = re.search(r"Expected (\d+)", data.get("error_message", ""))
    if not match:
        raise TranscriptionError(f"could not detect frame size, got: {data}")
    return int(match.group(1))


async def transcribe_chunk(path, call_id, frame_size_bytes, flush_timeout_s=None):
    """Transcribe one audio chunk file. Returns (transcript_text, duration_s).

    Raises TranscriptionTimeout if the server never replies after flush (a
    per-chunk quirk seen on some short/quiet chunks - safe to skip and continue).
    Raises TranscriptionError on other failures (audio load, connection, a
    server-side error message, or config_ack never arriving) - these indicate a
    real service problem, so the caller should stop and resume later.
    """
    flush_timeout_s = flush_timeout_s or config.FLUSH_RECV_TIMEOUT_S
    try:
        pcm_bytes, duration_s = load_pcm16(path, config.TARGET_SAMPLE_RATE)
    except Exception as e:
        raise TranscriptionError(f"audio load failed for {path}: {e}") from e

    min_dur, max_dur = _clamp_chunk_bounds(duration_s)
    uri = f"{config.LIT_WS_BASE}/{call_id}?language={config.LANGUAGE}"
    texts = []

    try:
        async with websockets.connect(uri, open_timeout=config.CONNECT_TIMEOUT_S) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "config",
                        "buffer": {
                            "min_chunk_duration_s": min_dur,
                            "max_chunk_duration_s": max_dur,
                        },
                        "transcriber": {"language": config.LANGUAGE, "temperature": 0.0},
                    }
                )
            )
            await asyncio.wait_for(ws.recv(), timeout=config.RECV_TIMEOUT_S)  # config_ack

            for seq, frame in enumerate(_split_frames(pcm_bytes, frame_size_bytes)):
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio_frame",
                            "call_id": call_id,
                            "sequence": seq,
                            "timestamp": int(time.time() * 1000),
                            "audio": base64.b64encode(frame).decode(),
                            "sample_rate": config.TARGET_SAMPLE_RATE,
                        }
                    )
                )
            await ws.send(json.dumps({"type": "flush"}))

            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=flush_timeout_s)
                    data = json.loads(raw)
                    msg_type = data.get("type")
                    if msg_type == "transcript":
                        texts.append(data["text"])
                        if data.get("is_final", True):
                            break
                    elif msg_type == "error":
                        raise TranscriptionError(
                            f"server error {data.get('error_code')}: {data.get('error_message')}"
                        )
                    # other message types (e.g. human_speaking) are ignored
            except asyncio.TimeoutError as e:
                raise TranscriptionTimeout(
                    f"no response after flush for {path} (waited {flush_timeout_s}s)"
                ) from e
    except TranscriptionError:
        raise
    except (asyncio.TimeoutError, websockets.exceptions.WebSocketException, OSError) as e:
        raise TranscriptionError(f"transcription failed for {path}: {e}") from e

    return " ".join(t.strip() for t in texts).strip(), duration_s


class LITTranscriber:
    """Default transcription backend: the streaming LIT websocket service."""

    name = "lit"

    def __init__(self):
        self.frame_size_bytes = None

    async def prepare(self):
        print("[transcribe] probing LIT server for frame size...")
        self.frame_size_bytes = await detect_frame_size_bytes()
        print(f"[transcribe] frame_size_bytes={self.frame_size_bytes}")

    async def transcribe_chunk(self, path, call_id, flush_timeout_s=None):
        return await transcribe_chunk(path, call_id, self.frame_size_bytes, flush_timeout_s=flush_timeout_s)


def build_transcriber(use_vllm=False):
    if use_vllm:
        from .transcribe_vllm import VLLMTranscriber

        return VLLMTranscriber()
    return LITTranscriber()
