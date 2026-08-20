import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the trained Lightning checkpoint (.ckpt) to load for inference.
# The old run1_checkpoints/ (41-intent taxonomy) are structurally incompatible
# with the current model - update this to your real v3 checkpoint once ready.
MODEL_PATH = os.path.join(
    THIS_DIR, "models", "v5/checkpoints", "whisper-small-S2I-hi-v5-epochepoch=02.ckpt"
)
MODEL_TYPE = "small"
INTENT_MAP_PATH = os.path.join(THIS_DIR, "models", "intent_to_idx.json")

DEVICE = "cuda"  # falls back to cpu automatically if unavailable

# --- Audio / VAD settings ---
TARGET_SAMPLE_RATE = 16000  # what the model expects (whisper.audio.SAMPLE_RATE)

# Simple energy-based VAD (no extra model/download needed). A chunk counts as
# "speech" if its dBFS is above this threshold.
VAD_DB_THRESHOLD = -40.0
# Keep "speech active" this long after the last loud chunk before considering
# the utterance over - smooths out brief dips mid-word so it doesn't flicker.
VAD_HANGOVER_MS = 400
# Once speech has stopped, wait this long before actually running
# classification (must be >= VAD_HANGOVER_MS to ever trigger).
SILENCE_TO_TRIGGER_MS = 500
# Ignore utterances shorter than this (likely a cough/click, not real speech).
MIN_UTTERANCE_MS = 250
# Force-classify if a single utterance runs longer than this without a pause.
MAX_UTTERANCE_MS = 8000

# How often the browser sends a new audio chunk to the backend.
STREAM_EVERY_S = 0.3

# How many past classifications to keep in the on-screen history table
# (newest pushed to the top, oldest popped off once this is exceeded).
HISTORY_LENGTH = 5

SERVER_PORT = 7860

# --- Audio folder review tab ---
# Directory of audio files to browse, predict on, and correct. Leave as None
# (or point it at a missing/empty folder) and that section just shows an
# empty state - nothing breaks.
AUDIO_FOLDER_PATH = '/mnt/HDD8TB/aman_ws/stt/data/example_audios'
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")

# Corrections get appended here, one JSON record per line.
FEEDBACK_LOG_PATH = os.path.join(THIS_DIR, "feedback", "corrections.jsonl")

# How many audio rows to render at once (each row has a live audio player -
# rendering hundreds simultaneously is slow/heavy in the browser).
AUDIO_LIST_PAGE_SIZE = 20
