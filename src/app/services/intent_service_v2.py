"""v2 inference service: same request/response contract as v1
(intent_service.py), but for the 15-intent model2 architecture, which was
trained with masked-mean pooling over only the real (non-padding) audio
frames (see intent-training/dataset2.py on the expt/all branch).

This is NOT optional to get right: model2's forward() silently accepts a
plain mel spectrogram with no valid_lengths and falls back to unmasked
pooling - no error, no shape mismatch, just measurably worse predictions
(~4x accuracy drop measured during eval on this exact checkpoint family, see
eval3.py's masked_pooling fix on expt/all). So unlike v1, this service must
compute and pass valid_lengths on every inference call.

Kept as a fully separate class/module from IntentService (rather than adding
an `if model_module == "model2"` branch there) so v1 stays completely
unaffected by anything here.
"""

from __future__ import annotations

import dataclasses
import io
import json
from contextlib import nullcontext

import av
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from starlette.concurrency import run_in_threadpool
from whisper.audio import HOP_LENGTH, N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim

from ..config import (
    DEVICE,
    MAX_AUDIO_DURATION_SECONDS,
    MIN_AUDIO_DURATION_SECONDS,
    MODEL_TYPE,
    TARGET_SAMPLE_RATE,
    logger,
    settings,
)
from .batching import DynamicBatcher
from .whisper_model_v2 import WhisperIntentClassification

# Matches dataset2.py's DURATION_CAP_S exactly - this is what the checkpoint
# was trained with (real audio content beyond this many seconds truncated
# before padding to the fixed 30s encoder window), not to be confused with
# MAX_AUDIO_DURATION_SECONDS (a request-validation bound, config/constants.py).
DURATION_CAP_S = 10.0
ENCODER_FRAMES_PER_SECOND = 50  # whisper encoder: 20ms/timestep
N_ENCODER_FRAMES = N_SAMPLES // HOP_LENGTH // 2  # 1500 for the standard 30s window


@dataclasses.dataclass
class _InferenceRequest:
    mel: torch.Tensor  # [1, n_mels, N_FRAMES], already on self.device
    valid_len: int  # real (non-padding) encoder timesteps, out of N_ENCODER_FRAMES
    top_k: int


class IntentServiceV2:
    """Loads the 15-intent model2 model and runs masked-pooling inference."""

    def __init__(self):
        self.device = "cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu"

        self.model = None
        self.idx_to_intent: dict[int, str] = {}
        self.intent_to_idx: dict[str, int] = {}
        self.num_classes: int = 0
        self.model_output_classes: int | None = None
        self.model_loaded: bool = False
        self._batcher: DynamicBatcher[_InferenceRequest, list[dict]] | None = None

    # -------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------

    def load(self):
        """Load config.json and model.bin from Hugging Face."""
        logger.info("=" * 80)
        logger.info("Starting intent model loading (v2)")
        logger.info("Device: %s", self.device)
        logger.info("MODEL_TYPE: %s", MODEL_TYPE)
        logger.info("HF_MODEL_REPO_V2: %s", settings.HF_MODEL_REPO_V2)
        logger.info("=" * 80)

        hf_repo = settings.HF_MODEL_REPO_V2
        if not hf_repo:
            raise RuntimeError("HF_MODEL_REPO_V2 is not configured")

        hf_token = settings.HF_TOKEN
        model_type = MODEL_TYPE
        if not model_type:
            raise RuntimeError("MODEL_TYPE is not configured")

        config_path = self._download_hf_file(hf_repo, "config.json", hf_token)
        model_path = self._download_hf_file(hf_repo, "model.bin", hf_token)

        model_config = self._read_config(config_path)
        self._check_model_module(model_config)
        self._load_intent_mapping(model_config)
        self._log_intent_mapping()

        model = self._build_model(model_type)
        state_dict = self._load_checkpoint(model_path)
        self._load_state_dict(model, state_dict, model_type)

        self._finalize_model(model, hf_repo, model_type)

    @staticmethod
    def _download_hf_file(hf_repo: str, filename: str, hf_token: str | None) -> str:
        try:
            logger.info("Downloading %s from Hugging Face: %s", filename, hf_repo)
            path = hf_hub_download(repo_id=hf_repo, filename=filename, token=hf_token)
            logger.info("%s downloaded: %s", filename, path)
            return path
        except Exception as exc:
            logger.exception("Failed to download %s from Hugging Face", filename)
            raise RuntimeError(
                f"Failed to download {filename} from HF repo '{hf_repo}': {exc}"
            ) from exc

    @staticmethod
    def _read_config(config_path: str) -> dict:
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                model_config = json.load(handle)
        except Exception as exc:
            logger.exception("Failed to parse config.json")
            raise RuntimeError(f"Unable to parse config.json: {exc}") from exc

        if not isinstance(model_config, dict):
            raise RuntimeError("config.json must contain a JSON object")
        if "intent_to_idx" not in model_config:
            raise RuntimeError("config.json is missing 'intent_to_idx'")
        if "idx_to_intent" not in model_config:
            raise RuntimeError("config.json is missing 'idx_to_intent'")

        return model_config

    @staticmethod
    def _check_model_module(model_config: dict):
        """Guard against pointing HF_MODEL_REPO_V2 at a v1-style (model.py)
        checkpoint by accident - that would load successfully here (same
        state_dict key names for the encoder) but silently mispredict, since
        this service always calls forward() with valid_lengths."""
        model_module = model_config.get("model_module")
        if model_module is not None and model_module != "model2":
            raise RuntimeError(
                f"config.json declares model_module={model_module!r}, but the v2 service "
                "only supports 'model2' checkpoints (masked pooling). Point HF_MODEL_REPO_V2 "
                "at a checkpoint pushed with --model-module model2."
            )

    def _load_intent_mapping(self, model_config: dict):
        raw_intent_to_idx = model_config["intent_to_idx"]
        raw_idx_to_intent = model_config["idx_to_intent"]

        if not isinstance(raw_intent_to_idx, dict):
            raise RuntimeError("'intent_to_idx' must be a JSON object")
        if not isinstance(raw_idx_to_intent, dict):
            raise RuntimeError("'idx_to_intent' must be a JSON object")

        try:
            self.intent_to_idx = {str(k): int(v) for k, v in raw_intent_to_idx.items()}
        except Exception as exc:
            raise RuntimeError(
                f"Invalid 'intent_to_idx' in config.json. Could not convert class indices to integers: {exc}"
            ) from exc

        try:
            self.idx_to_intent = {int(k): str(v) for k, v in raw_idx_to_intent.items()}
        except Exception as exc:
            raise RuntimeError(
                f"Invalid 'idx_to_intent' in config.json. Could not convert indices to integers: {exc}"
            ) from exc

        self.num_classes = len(self.intent_to_idx)
        if self.num_classes <= 0:
            raise RuntimeError("No intents found in intent_to_idx")

        expected_indices = list(range(self.num_classes))
        intent_indices = sorted(self.intent_to_idx.values())
        reverse_indices = sorted(self.idx_to_intent.keys())

        if intent_indices != expected_indices or reverse_indices != expected_indices:
            raise RuntimeError("Non-contiguous indices in intent mapping config.")

        for intent, index in self.intent_to_idx.items():
            reverse_intent = self.idx_to_intent.get(index)
            if reverse_intent != intent:
                raise RuntimeError(f"Intent mapping mismatch for '{intent}'")

    def _log_intent_mapping(self):
        logger.info("Loaded intent mapping with %d classes", self.num_classes)
        for index in range(self.num_classes):
            logger.info("  class[%d] -> %s", index, self.idx_to_intent[index])

    def _build_model(self, model_type: str):
        logger.info("Creating v2 WhisperIntentClassification with n_class=%d", self.num_classes)
        try:
            return WhisperIntentClassification(model_type, n_class=self.num_classes)
        except Exception as exc:
            logger.exception("Failed to instantiate v2 WhisperIntentClassification")
            raise RuntimeError(f"Failed to create v2 WhisperIntentClassification: {exc}") from exc

    def _load_checkpoint(self, model_path: str) -> dict:
        logger.info("Loading checkpoint: %s", model_path)
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as exc:
            logger.exception("Failed to load model.bin")
            raise RuntimeError(f"Unable to load model checkpoint: {exc}") from exc

        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        if not isinstance(state_dict, dict):
            raise RuntimeError("Model checkpoint does not contain a valid state_dict")

        has_model_prefix = any(str(k).startswith("model.") for k in state_dict.keys())
        if has_model_prefix:
            logger.info("Detected 'model.' prefix in checkpoint keys; stripping it")
            state_dict = {
                k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")
            }

        return state_dict

    def _load_state_dict(self, model, state_dict: dict, model_type: str):
        try:
            incompatible = model.load_state_dict(state_dict, strict=True)
            logger.info("Model state_dict loaded successfully")
            if incompatible.missing_keys:
                logger.error("Missing keys: %s", incompatible.missing_keys)
            if incompatible.unexpected_keys:
                logger.error("Unexpected keys: %s", incompatible.unexpected_keys)
        except Exception as exc:
            logger.exception("FAILED to load model state_dict")
            raise RuntimeError(
                f"Model architecture/checkpoint mismatch. Original error: {exc}"
            ) from exc

    def _finalize_model(self, model, hf_repo: str, model_type: str):
        try:
            model.to(self.device)
            model.eval()
        except Exception as exc:
            logger.exception("Failed to move model to device")
            raise RuntimeError(f"Failed to move model to {self.device}: {exc}") from exc

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA enabled: %s", torch.cuda.get_device_name(0))

            # No torch.compile here (unlike v1) - model2's forward() takes a
            # conditional branch on valid_lengths being None/not-None, which
            # is fine for compile in principle, but this service never calls
            # it in the None branch, so there's nothing to gain by compiling
            # a code path that's never exercised. Can be added later if
            # warmup/steady-state latency needs it.

            logger.info("Performing CUDA model warmup pass...")
            dummy_mel = torch.zeros((1, 80, 3000), device=self.device)
            dummy_valid_lengths = torch.tensor([N_ENCODER_FRAMES], device=self.device)
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
            with torch.inference_mode(), autocast_context:
                _ = model(dummy_mel, valid_lengths=dummy_valid_lengths)
            logger.info("CUDA warmup completed successfully")

        self.model = model
        self.model_loaded = True

        self._batcher = DynamicBatcher(
            self._run_batch_inference,
            max_batch_size=settings.BATCH_MAX_SIZE,
            max_wait_s=settings.BATCH_MAX_WAIT_MS / 1000.0,
        )
        self._batcher.start()

        logger.info("=" * 80)
        logger.info("MODEL LOADED SUCCESSFULLY (v2)")
        logger.info("Model repo: %s", hf_repo)
        logger.info("Model type: %s", model_type)
        logger.info("Number of intents: %d", self.num_classes)
        logger.info("Device: %s", self.device)
        logger.info(
            "Batching: max_batch_size=%d max_wait_ms=%.1f",
            settings.BATCH_MAX_SIZE, settings.BATCH_MAX_WAIT_MS,
        )
        logger.info("=" * 80)

    async def shutdown(self):
        if self._batcher is not None:
            await self._batcher.stop()
            self._batcher = None

    # -------------------------------------------------------------------
    # Audio Prediction
    # -------------------------------------------------------------------

    async def predict(self, payload: bytes, suffix: str, top_k: int):
        if self.model is None or self._batcher is None:
            raise RuntimeError("Model is not loaded")

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, received {top_k}")

        if top_k > self.num_classes:
            logger.warning(
                "Requested top_k=%d but model has only %d classes. Clamping top_k to %d.",
                top_k, self.num_classes, self.num_classes,
            )
            top_k = self.num_classes

        mel, valid_len, duration = await run_in_threadpool(self._decode_and_preprocess, payload)
        ranked_intents = await self._batcher.submit(_InferenceRequest(mel=mel, valid_len=valid_len, top_k=top_k))

        logger.info(
            "Prediction: %s | duration=%.3fs | valid_len=%d | top_k=%d",
            ranked_intents[0] if ranked_intents else None, duration, valid_len, top_k,
        )

        return ranked_intents, duration

    def _decode_and_preprocess(self, payload: bytes) -> tuple[torch.Tensor, int, float]:
        audio = self._decode_audio(payload)
        duration = self._validate_duration(audio)
        mel, valid_len = self._preprocess_audio(audio)
        return mel, valid_len, duration

    @staticmethod
    def _decode_audio(payload: bytes) -> np.ndarray:
        try:
            with av.open(io.BytesIO(payload)) as container:
                stream = next(s for s in container.streams if s.type == "audio")
                resampler = av.AudioResampler(
                    format="s16p", layout="mono", rate=TARGET_SAMPLE_RATE
                )

                audio_frames = []
                for frame in container.decode(stream):
                    audio_frames.extend(resampler.resample(frame))

                if not audio_frames:
                    raise ValueError("Decoded audio container yielded no audio frames.")

                audio = np.concatenate(
                    [f.to_ndarray().flatten() for f in audio_frames]
                ).astype(np.float32) / 32768.0

                return audio

        except Exception as exc:
            logger.exception("In-memory audio decoding failed")
            raise ValueError("Unable to decode the uploaded audio.") from exc

    @staticmethod
    def _validate_duration(audio: np.ndarray) -> float:
        duration = len(audio) / TARGET_SAMPLE_RATE
        logger.debug("Audio duration: %.3f seconds", duration)

        if not (MIN_AUDIO_DURATION_SECONDS <= duration <= MAX_AUDIO_DURATION_SECONDS):
            raise ValueError(
                f"Audio duration must be between {MIN_AUDIO_DURATION_SECONDS:g} and "
                f"{MAX_AUDIO_DURATION_SECONDS:g} seconds (received {duration:.3f} seconds)."
            )
        return duration

    def _preprocess_audio(self, audio: np.ndarray) -> tuple[torch.Tensor, int]:
        """Mirrors dataset2.py's HFIntentDataset.__getitem__ exactly: cap real
        content at DURATION_CAP_S, compute valid_len from that, THEN pad/trim
        to the full 30s window for the mel spectrogram."""
        try:
            max_samples = int(DURATION_CAP_S * TARGET_SAMPLE_RATE)
            real_samples = min(len(audio), max_samples)
            capped_audio = audio[:real_samples]
            valid_len = min(
                N_ENCODER_FRAMES,
                max(1, round(real_samples / TARGET_SAMPLE_RATE * ENCODER_FRAMES_PER_SECOND)),
            )

            audio_tensor = torch.from_numpy(capped_audio).to(self.device)
            samples = pad_or_trim(audio_tensor, N_SAMPLES)
            mel = log_mel_spectrogram(samples).unsqueeze(0)
            return mel, valid_len
        except Exception as exc:
            logger.exception("Audio preprocessing failed")
            raise RuntimeError(f"Audio preprocessing failed: {exc}") from exc

    def _run_batch_inference(self, requests: list[_InferenceRequest]) -> list[list[dict]]:
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device == "cuda" else nullcontext()
        )

        batch_size = len(requests)

        try:
            with torch.inference_mode(), autocast_context:
                mels = torch.cat([r.mel for r in requests], dim=0)
                valid_lengths = torch.tensor(
                    [r.valid_len for r in requests], device=self.device, dtype=torch.long,
                )
                logits = self.model(mels, valid_lengths=valid_lengths)

                if logits.ndim != 2 or logits.shape[0] != batch_size:
                    raise RuntimeError(f"Unexpected output shape: {tuple(logits.shape)}")

                output_classes = int(logits.shape[1])
                self.model_output_classes = output_classes

                if output_classes != self.num_classes:
                    raise RuntimeError("Model output shape mismatch with configured intents")

                probabilities = torch.softmax(logits, dim=1)

                results = []
                for i, request in enumerate(requests):
                    scores, indices = torch.topk(probabilities[i], k=request.top_k)
                    results.append(self._to_ranked_intents(scores, indices))
                return results

        except Exception as exc:
            logger.exception("MODEL INFERENCE FAILED (batch_size=%d)", batch_size)
            raise RuntimeError(f"Model inference failed: {exc}") from exc

    def _to_ranked_intents(self, scores: torch.Tensor, indices: torch.Tensor) -> list[dict]:
        ranked_intents = []
        for score, index in zip(scores.detach().cpu().tolist(), indices.detach().cpu().tolist()):
            index = int(index)
            if index not in self.idx_to_intent:
                raise RuntimeError(f"Index {index} missing from mapping")
            ranked_intents.append({"intent": self.idx_to_intent[index], "confidence": float(score)})
        return ranked_intents


service_v2 = IntentServiceV2()
