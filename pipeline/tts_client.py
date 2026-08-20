"""TTS synthesis client - adapted from the endpoint owner's own synth() script.

Reads TTS_WS_ENDPOINT (and optionally TTS_VOICE_ID) from .env. TTS_WS_ENDPOINT
should be the base ws:// URL up to the path prefix, e.g. "ws://host:port/tts" -
this module appends "/ws/{call_id}" itself.
"""

import asyncio
import json
import uuid
import wave
from pathlib import Path

import websockets

from . import config


def _split(raw):
    """Split a combined binary frame ({json header} + raw PCM) -> (msg, pcm_bytes)."""
    if isinstance(raw, str):
        return json.loads(raw), b""
    depth = end = 0
    for i, b in enumerate(raw):
        if b == 0x7B:  # '{'
            depth += 1
        elif b == 0x7D:  # '}'
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return json.loads(raw[:end]), raw[end:]


async def synth(text, out_path, voice=None, lang=None, speed=None):
    """Synthesize `text`, write a wav to out_path. Returns (pcm_bytes, sample_rate)."""
    call_id = f"nb-{uuid.uuid4().hex[:8]}"
    url = f"{config.TTS_WS_ENDPOINT.rstrip('/')}/{call_id}"

    pcm, sr, saw_final, done = bytearray(), 24000, False, {}
    req = {
        "type": "synthesize",
        "call_id": call_id,
        "text_id": uuid.uuid4().hex[:8],
        "text": text,
        "streaming": True,
    }
    voice = voice or config.TTS_VOICE_ID
    if voice:
        req["voice_id"] = voice
    if lang:
        req["language"] = lang
    if speed:
        req["speed"] = speed

    async with websockets.connect(
        url, max_size=100 * 1024 * 1024, open_timeout=10, close_timeout=3, ping_interval=None
    ) as ws:
        await ws.send(json.dumps(req, ensure_ascii=False))
        while True:
            try:
                raw = await asyncio.wait_for(
                    ws.recv(), timeout=(5 if saw_final else config.TTS_TIMEOUT_SECONDS)
                )
            except asyncio.TimeoutError:
                if saw_final:
                    break
                raise RuntimeError("TIMEOUT waiting for audio")
            msg, audio = _split(raw)
            mt = msg.get("type")
            if mt == "audio_chunk":
                if not audio:
                    audio = await ws.recv()
                    if isinstance(audio, str):
                        audio = audio.encode()
                sr = msg.get("sample_rate", sr)
                pcm += audio
                if msg.get("is_final"):
                    saw_final = True
            elif mt == "audio_done":
                done = msg
                break
            elif mt == "error":
                raise RuntimeError(msg.get("error"))

    if not pcm:
        raise RuntimeError("empty audio returned")

    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(pcm))
    return bytes(pcm), sr


async def tts_once(text, out_path: Path, voice=None, speed=None):
    await synth(text, out_path, voice=voice, speed=speed)


async def tts_with_retry(record_id, text, out_path: Path, sem: asyncio.Semaphore, voice=None, speed=None):
    async with sem:
        tmp_path = out_path.with_suffix(".wav.tmp")
        last_exc = None
        for attempt in range(1, config.TTS_MAX_RETRIES + 1):
            try:
                await tts_once(text, tmp_path, voice=voice, speed=speed)
                tmp_path.rename(out_path)
                return record_id, True, None
            except Exception as e:  # noqa: BLE001
                last_exc = e
                await asyncio.sleep(min(2**attempt, 10))
        tmp_path.unlink(missing_ok=True)
        return record_id, False, str(last_exc)
