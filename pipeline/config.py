import os

from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


# Loads OPENAI_API_KEY (and any other overrides) from .env in the project root,
# if that file exists. Values already set in the environment take precedence.
load_dotenv(os.path.join(BASE_DIR, "pipeline/.env"))

SOURCE_DATASET_DIR = os.environ.get("SOURCE_DATASET_DIR", "")

DATA_DIR = '/home/jovyan/aman_ws/stt/LIT-intent-training/data'
OUTPUT_DIR = os.path.join(DATA_DIR, "call_trascript_intent_data")
SAMPLE_OUTPUT_DIR = os.path.join(DATA_DIR, "call_trascript_intent_data_sample")
# Second generation pass (e.g. with a different transcription backend) - kept
# separate so it can be appended to the already-pushed v1 dataset later rather
# than mixed into it. Point `run --exclude-dir OUTPUT_DIR` at v1 so v2 only
# ever processes chunks v1 hasn't already finished.
OUTPUT_DIR_V2 = os.path.join(DATA_DIR, "call_trascript_intent_data_v2")
# v3: full re-classification of v1+v2's existing transcripts under the new,
# consolidated 20-intent taxonomy - no new transcription, just a fresh intent
# label per row (see pipeline/build_v3.py).
OUTPUT_DIR_V3 = os.path.join(DATA_DIR, "call_trascript_intent_data_v3")

INTENTS_MD_PATH = os.path.join(BASE_DIR, "intents.md")
# Auto-generated cache of the parsed taxonomy - intents.md is only read once,
# the first time this doesn't exist. Delete this file to force a re-parse
# (e.g. after editing intents.md).
INTENTS_JSON_PATH = os.path.join(BASE_DIR, "intents.json")
GEMINI_KEY_PATH = os.path.join(BASE_DIR, "kapture-gemini-9d92c74fb501.json")
# Kept in sync with intents/data by relabel_unclear.py --apply (also
# refreshable any time by re-running pipeline.intent_analytics by hand).
INTENT_ANALYTICS_PATH = os.path.join(BASE_DIR, "intent_analytics.json")

DATA_FILENAME = "data.jsonl"
METADATA_FILENAME = "metadata.json"
ERRORS_LOG_FILENAME = "errors.log"
SKIPPED_FILENAME = "skipped.jsonl"
# Transcribed chunks waiting for a batch intent-classification call. Durable so a
# stop mid-batch doesn't lose already-completed transcription work.
PENDING_INTENT_FILENAME = "pending_intent.jsonl"

# Sample rate of the source chunk wav files (used to convert the source dataset's
# sample-count timestamps into milliseconds).
SOURCE_SAMPLE_RATE = 8000

# Transcription (LIT websocket) settings.
LIT_WS_BASE = os.environ.get("LIT_WS_BASE", "ws://localhost:4000/lit/ws")
LANGUAGE = "hi"
TARGET_SAMPLE_RATE = 16000
CONNECT_TIMEOUT_S = 5.0
RECV_TIMEOUT_S = 5.0
# Shorter timeout specifically for waiting on a transcript after `flush`. Some
# short/quiet chunks never get a reply (server-side quirk, not an outage) - keep
# this short so those get skipped quickly instead of stalling the whole pipeline.
FLUSH_RECV_TIMEOUT_S = 10.0
# Longer timeout used when explicitly retrying previously-skipped chunks.
RETRY_FLUSH_RECV_TIMEOUT_S = 10.0

# Gentle pacing between chunks to avoid hammering the STT/LLM services.
INTER_CHUNK_SLEEP_S = 0.1

# Transcription failures (hangs, connection drops, DNS blips, server errors) are
# skipped and the run continues. But this many IN A ROW suggests a real outage
# rather than scattered per-chunk noise, so stop instead of skipping everything.
MAX_CONSECUTIVE_TRANSCRIPTION_FAILURES = 25

# Gemini (Vertex AI) settings. Default intent-classification backend.
GEMINI_PROJECT = "kapture-gemini"
GEMINI_LOCATION = "us-central1"
GEMINI_MODEL = "gemini-2.5-flash"

# OpenAI settings. Alternative backend, selected with --use-openai. Reads
# OPENAI_API_KEY from the environment / .env (see load_dotenv above).
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Intent classification is done in batches (one LLM call classifies many
# transcripts at once) instead of one call per chunk, to cut down request count.
# Gemini's structured-output parsing fails more often at 100 - 50 is more reliable.
INTENT_BATCH_SIZE = 50

# A batch classification call occasionally fails to parse (truncated/malformed
# structured output) - retry it this many times before giving up and stopping.
CLASSIFY_MAX_RETRIES = 2
CLASSIFY_RETRY_DELAY_S = 3.0

# Hugging Face Hub push settings. Reads from .env - see load_dotenv above.
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID")

HF_DATASET_REPO = "kapturecx/call-transcript-intent-data"

HF_DATASET_SPLITS = (
    "train",
    "validation",
    "eval",
)

RELABEL_BATCH_SIZE = 20
RELABEL_MAX_CONCURRENCY = 10
RELABEL_MAX_RETRIES = 4
RELABEL_RETRY_DELAY_S = 2.0

RELABEL_TEST_OUTPUT = "test_relabel.json"
RELABEL_FULL_OUTPUT = "call-transcript-intent-data-relabeled"
RELABEL_CHECKPOINT = "relabel_checkpoint.jsonl"

RELABEL_HF_AUDIO_PREFIX = "audio"

RELABEL_PUSH_AUDIO_BATCH_SIZE = 500

RELABEL_HF_PRIVATE = False

# vLLM transcription backend, selected with --use-vllm. Standard OpenAI-compatible
# /v1/audio/transcriptions endpoint. Two ways to reach it:
#   - local: `vllm serve kapturecx/qwen-asr-hindi-3006-ft --max-model-len 768 \
#             --max-num-seqs 16 --port 5500 --gpu-memory-utilization 0.6`
#   - via ngrok tunnel to the same server, at NGROK_ENDPOINT + QWEN_ASR_PATH
# If NGROK_ENDPOINT is set in .env, VLLMTranscriber uses that automatically;
# otherwise it falls back to the local VLLM_BASE_URL - existing local-only setups
# keep working unchanged either way.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:5500")
VLLM_LOCAL_PATH = "/v1/audio/transcriptions"
NGROK_ENDPOINT = os.environ.get("NGROK_ENDPOINT")
if NGROK_ENDPOINT and not NGROK_ENDPOINT.startswith(("http://", "https://")):
    NGROK_ENDPOINT = "https://" + NGROK_ENDPOINT  # .env commonly has just the hostname
QWEN_ASR_PATH = "/qwen-asr/v1/audio/transcriptions"
VLLM_MODEL = os.environ.get("VLLM_MODEL", "kapturecx/qwen-asr-hindi-3006-ft")
VLLM_TIMEOUT_S = 30.0
# ngrok's free plan documents two SEPARATE rate limits: 100 new TCP
# connections/min, and 4000 HTTP requests/min (https://ngrok.com/docs/pricing-limits/free-plan-limits).
# VLLMTranscriber reuses one persistent connection (see requests.Session()
# below) so it mostly isn't opening new TCP connections at all - this caps the
# HTTP request rate itself, well under ngrok's 4000/min ceiling, as a safety net.
VLLM_MAX_REQUESTS_PER_MINUTE = 2000

# TTS synthesis, for augmenting underrepresented intents. Reads from .env.
# TTS_WS_ENDPOINT is the full base URL INCLUDING any path prefix the server
# expects (e.g. ".../ws") - pipeline/tts_client.py appends only "/{call_id}".
# TTS_VOICE_ID is a fallback default; voice-cloned users get their own voice_id
# (see voice_clone_users.json) once that pipeline is wired in.
TTS_WS_ENDPOINT = os.environ.get("TTS_WS_ENDPOINT")
if TTS_WS_ENDPOINT and TTS_WS_ENDPOINT.startswith("http"):
    TTS_WS_ENDPOINT = "ws" + TTS_WS_ENDPOINT[4:]  # http(s):// -> ws(s)://
TTS_VOICE_ID = os.environ.get("TTS_VOICE_ID", "")
TTS_TIMEOUT_SECONDS = 30.0
TTS_MAX_RETRIES = 3
TTS_CONCURRENCY = 10

# Augmentation for underrepresented intents (see pipeline/augment_underrepresented.py).
# For EVERY intent: new_needed = max(AUGMENT_MIN_NEW_PER_INTENT, max(AUGMENT_TARGET_TOTAL - current_total, 0))
# where current_total = real v3 rows + already-synthesized rows for that intent.
AUGMENT_MIN_NEW_PER_INTENT = 500
AUGMENT_TARGET_TOTAL = 4500
AUGMENT_SENTENCE_BATCH_SIZE = 100
AUGMENT_MODEL = "gpt-5.6-luna"
OUTPUT_DIR_AUGMENTED = os.path.join(DATA_DIR, "call_trascript_intent_data_augmented")

# Random per-sample TTS speed-up so augmented audio isn't uniformly the same
# pace as the reference voice clip. 1.0 = server default (measured as
# noticeably slow for our voices); randomized up to 1.7x for natural variety.
AUGMENT_SPEED_MIN = 1.0
AUGMENT_SPEED_MAX = 1.7

# Voice-cloning sample curation (see pipeline/voice_clone_candidates.py and
# pipeline/voice_clone_ui.py).
VOICE_CLONE_USERS_JSON = os.path.join(BASE_DIR, "voice_clone_users.json")
VC_SAMPLES_DIR = os.path.join(DATA_DIR, "VC_samples")
VC_CANDIDATES_PATH = os.path.join(DATA_DIR, "voice_clone_candidates.json")
VC_REJECTED_PATH = os.path.join(DATA_DIR, "voice_clone_rejected.json")
VC_MIN_TRANSCRIPT_CHARS = 20
VC_MIN_DURATION_S = 3.0
VC_DEFAULT_CLIP_SECONDS = 5.0
