# Intent Classification API

OpenAI-compatible Whisper intent-classification service. Self-contained -
no dependency on anything else in this repository.

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` on `PATH` (used by Whisper to decode uploaded audio)
- A CUDA GPU for production use (falls back to CPU automatically if
  `DEVICE == "cuda"` but none is available - see
  [app/config/constants.py](app/config/constants.py))

## Setup

```bash
cd src
uv sync
cp .env.example .env
```

Edit `.env` and set at least `HF_TOKEN` if the model repo in
`HF_MODEL_REPO` is private. `HF_MODEL_REPO` defaults to
`amn-raw/whisper-small-intent17-classifier`; override it in `.env` to
serve a different model. The repo must contain `config.json` (with
`intent_to_idx` / `idx_to_intent`) and `model.bin`.

## Run

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 4000 --workers 1
```

or

```bash
uv run python -m app.main
```

On startup the service downloads the model from Hugging Face and loads it
onto `DEVICE` (falling back to CPU if CUDA isn't available). If loading fails, the server still starts so
`/health` can report why, instead of crashing outright.

### Scaling / concurrency

Concurrent requests share batched GPU forward passes instead of each doing
its own batch-of-1 call (see `DynamicBatcher` in
[app/services/batching.py](app/services/batching.py)) - audio
decode/preprocessing (CPU-bound: spawns `ffmpeg`, computes a mel
spectrogram) runs independently in a thread pool, so it doesn't block on or
serialize with the GPU step.

Because of this, **prefer one worker process per GPU, not many** - each
worker loads its own full copy of the model, and extra worker processes on
the same GPU just add memory pressure and CUDA context-switching overhead
without adding real throughput (batching within a single worker already
gives you the concurrency). Only add workers if you have multiple GPUs (one
worker per GPU) or you're CPU/decode-bound rather than GPU-bound.

Tune via `.env`:

- `BATCH_MAX_SIZE` (default `16`) / `BATCH_MAX_WAIT_MS` (default `10`) - how
  many concurrent requests can join one GPU forward pass, and how long the
  first one waits for others before running as-is. Raise `BATCH_MAX_SIZE`
  and/or `BATCH_MAX_WAIT_MS` for more throughput at the cost of per-request
  latency; lower them for latency at the cost of throughput.
- `DECODE_THREAD_POOL_SIZE` (default: framework default, currently 40) - how
  many audio decodes can run concurrently. Raise this if CPU/`ffmpeg`
  decoding (not the GPU) turns out to be the bottleneck under load.

## Configuration

- [app/config/settings.py](app/config/settings.py) - values that differ per
  deployment (`HF_MODEL_REPO`, `HF_TOKEN`, `HOST`, `PORT`, `LOG_LEVEL`,
  `DEBUG_ERRORS`, `BATCH_MAX_SIZE`, `BATCH_MAX_WAIT_MS`,
  `DECODE_THREAD_POOL_SIZE`). Read from `.env` / the environment; env vars
  win.
- [app/config/constants.py](app/config/constants.py) - fixed, code-level
  defaults not meant to vary per deployment (model type, audio duration
  limits, upload size cap, max `top_k`).

## Endpoints

- `GET /health` - `200` with device/model info once the model is loaded,
  `503` with diagnostics otherwise.
- `POST /intent/v1/audio/transcriptions` - OpenAI audio-transcriptions-
  compatible; the returned "transcription" is the predicted intent.

```bash
curl http://localhost:8000/health

curl http://localhost:8000/intent/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F response_format=verbose_json \
  -F top_k=3
```

`response_format` is `json` (default), `verbose_json`, or `text`.

## Project layout

```
app/
  main.py           FastAPI app, lifespan (model load/unload), uvicorn entrypoint
  config/           settings.py (env-driven) + constants.py (fixed) + logging.py
  api/v1/           route handlers (routes.py) and response models (schemas.py)
  services/         IntentService (model loading + inference) and the Whisper model
```
