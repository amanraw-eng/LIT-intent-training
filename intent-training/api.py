import asyncio
import json
import os
import tempfile
import time
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
from fastapi import FastAPI, Form, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from whisper.audio import N_SAMPLES, load_audio, log_mel_spectrogram, pad_or_trim

import API_CONFIG as cfg
from model import WhisperIntentClassification

from huggingface_hub import hf_hub_download


# =====================================================================
# OpenAI-Compatible Protocols & Response Models
# =====================================================================

TranscriptionResponseFormat = Literal["json", "verbose_json", "text"]


class IntentScore(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)


class TranscriptionUsageAudio(BaseModel):
    type: Literal["duration"] = "duration"
    seconds: float


class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.0
    temperature: float = 0.0
    tokens: list[int] = Field(default_factory=list)
    seek: int = 0


# Standard OpenAI JSON Response
class TranscriptionResponse(BaseModel):
    text: str
    usage: TranscriptionUsageAudio | None = None


# OpenAI Verbose JSON Response (Extended to convey intent details)
class TranscriptionResponseVerbose(BaseModel):
    task: str = "transcribe"
    language: str = "en"
    duration: float
    text: str
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    top_intent: str
    confidence: float
    intents: list[IntentScore]
    device: str


class HealthResponse(BaseModel):
    status: str
    device: str
    model_type: str
    num_classes: int


# =====================================================================
# Inference Service Logic
# =====================================================================

class IntentService:
    def __init__(self):
        self.device = cfg.DEVICE if cfg.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = None
        self.idx_to_intent = {}
        self.inference_lock = asyncio.Lock()

    # def load(self):
    #     if not os.path.isfile(cfg.MODEL_PATH):
    #         raise RuntimeError(f"Model checkpoint not found: {cfg.MODEL_PATH}")
    #     if not os.path.isfile(cfg.INTENT_MAP_PATH):
    #         raise RuntimeError(f"Intent map not found: {cfg.INTENT_MAP_PATH}")

    #     with open(cfg.INTENT_MAP_PATH, "r", encoding="utf-8") as handle:
    #         intent_to_idx = json.load(handle)
    #     self.idx_to_intent = {int(index): intent for intent, index in intent_to_idx.items()}

    #     model = WhisperIntentClassification(cfg.MODEL_TYPE, n_class=len(intent_to_idx))
    #     checkpoint = torch.load(cfg.MODEL_PATH, map_location=self.device, weights_only=False)
    #     state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    #     state_dict = {
    #         key[len("model."):]: value
    #         for key, value in state_dict.items()
    #         if key.startswith("model.")
    #     }
    #     model.load_state_dict(state_dict, strict=True)
    #     model.to(self.device)
    #     model.eval()
    #     if self.device == "cuda":
    #         torch.backends.cudnn.benchmark = True
    #     self.model = model



    def load(self):
        # Download config.json and model.bin from Hugging Face Hub
        config_path = hf_hub_download(repo_id=cfg.HF_MODEL_REPO, filename="config.json", token=cfg.HF_TOKEN)
        model_path = hf_hub_download(repo_id=cfg.HF_MODEL_REPO, filename="model.bin", token=cfg.HF_TOKEN)

        # Load mapping from config.json
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)

        # Handle int conversion regardless of string/int keys in JSON
        self.idx_to_intent = {int(idx): intent for idx, intent in config["idx_to_intent"].items()}
        intent_to_idx = config["intent_to_idx"]

        # Instantiate model
        model = WhisperIntentClassification(cfg.MODEL_TYPE, n_class=len(intent_to_idx))

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

        # Strip prefix if present
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {
                key[len("model."):]: value
                for key, value in state_dict.items()
                if key.startswith("model.")
            }

        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = model

    def predict_file(self, payload, suffix, top_k):
        file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(payload)
                file_path = handle.name
            audio = load_audio(file_path, sr=cfg.TARGET_SAMPLE_RATE)
        except Exception as exc:
            raise ValueError("Unable to decode the uploaded audio.") from exc
        finally:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)

        duration = len(audio) / cfg.TARGET_SAMPLE_RATE
        if not cfg.MIN_AUDIO_DURATION_SECONDS <= duration <= cfg.MAX_AUDIO_DURATION_SECONDS:
            raise ValueError(
                "Audio duration must be between "
                f"{cfg.MIN_AUDIO_DURATION_SECONDS:g} and "
                f"{cfg.MAX_AUDIO_DURATION_SECONDS:g} seconds "
                f"(received {duration:.3f} seconds)."
            )

        samples = pad_or_trim(np.asarray(audio, dtype=np.float32), N_SAMPLES)
        mel = log_mel_spectrogram(samples).unsqueeze(0).to(self.device)

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            probabilities = torch.softmax(self.model(mel), dim=1)[0]
            scores, indices = torch.topk(probabilities, k=top_k)

        intents = [
            {"intent": self.idx_to_intent[int(index)], "confidence": float(score)}
            for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist())
        ]
        return intents, duration


service = IntentService()


@asynccontextmanager
async def lifespan(app):
    service.load()
    yield
    service.model = None
    if service.device == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(
    title="OpenAI-Compatible Whisper Intent Classification API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    if service.model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return HealthResponse(
        status="ok",
        device=service.device,
        model_type=cfg.MODEL_TYPE,
        num_classes=len(service.idx_to_intent),
    )


# =====================================================================
# OpenAI Audio Transcriptions Route
# =====================================================================

@app.post(
    "/intent/v1/audio/transcriptions",
    response_model=TranscriptionResponse | TranscriptionResponseVerbose | str,
)
async def create_transcription(
    file: Annotated[UploadFile, File(description="The audio file object to transcribe.")],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[TranscriptionResponseFormat, Form()] = "json",
    temperature: Annotated[float | None, Form()] = 0.0,
    top_k: Annotated[int | None, Form()] = None,
):
    """
    OpenAI-compatible speech transcription endpoint mapping intent classification
    to OpenAI audio API response structures.
    """
    if service.model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    payload = await file.read(cfg.MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not payload:
        raise HTTPException(status_code=422, detail="Audio upload is empty")
    if len(payload) > cfg.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio upload exceeds the {cfg.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    k_value = top_k if top_k is not None else getattr(cfg, "MAX_TOP_K", 5)

    try:
        async with service.inference_lock:
            ranked_intents, duration = await run_in_threadpool(
                service.predict_file, payload, suffix, k_value
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    top_intent = ranked_intents[0]["intent"]
    top_confidence = ranked_intents[0]["confidence"]

    # 1. Standard OpenAI Plain Text Response
    if response_format == "text":
        return top_intent

    # 2. OpenAI Verbose JSON Response (Provides full classification breakdown)
    if response_format == "verbose_json":
        return TranscriptionResponseVerbose(
            task="transcribe",
            language=language or "en",
            duration=round(duration, 3),
            text=top_intent,
            segments=[
                TranscriptionSegment(
                    id=0,
                    start=0.0,
                    end=round(duration, 3),
                    text=top_intent,
                )
            ],
            top_intent=top_intent,
            confidence=top_confidence,
            intents=[IntentScore(**item) for item in ranked_intents],
            device=service.device,
        )

    # 3. Standard OpenAI JSON Response (Default)
    return TranscriptionResponse(
        text=top_intent,
        usage=TranscriptionUsageAudio(type="duration", seconds=round(duration, 3)),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=cfg.HOST, port=cfg.PORT, workers=1)