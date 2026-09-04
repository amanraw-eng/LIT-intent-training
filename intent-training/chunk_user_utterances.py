import os
import json
from pathlib import Path
import torch
from dotenv import load_dotenv
load_dotenv('.env')
BASE_DIR = os.environ.get("BASE_DIR", "")

# Load Silero VAD Model and utility functions via PyTorch Hub
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False
)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

INPUT_DIR = "/mnt/HDD8TB/aman_ws/stt/LIT-intent-training/data/August_data/user_utterances"
OUTPUT_DIR = "/mnt/HDD8TB/aman_ws/stt/LIT-intent-training/data/August_data/user_chunks"
MANIFEST_FILE = "/mnt/HDD8TB/aman_ws/stt/LIT-intent-training/data/August_data/user_chunks_manifest.json"
TARGET_SAMPLE_RATE = 16000  # Silero VAD operates natively at 16000 Hz

os.makedirs(OUTPUT_DIR, exist_ok=True)
chunks_manifest = []

for file_name in os.listdir(INPUT_DIR):
    if not file_name.lower().endswith(('.wav', '.mp3', '.flac')):
        continue

    file_path = os.path.join(INPUT_DIR, file_name)
    base_name = Path(file_name).stem

    try:
        print(f"Running VAD on: {file_name}")
        
        # Load and automatically resample audio to 16kHz
        wav = read_audio(file_path, sampling_rate=TARGET_SAMPLE_RATE)

        # Detect speech timestamps
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=TARGET_SAMPLE_RATE,
            threshold=0.5,                  # Speech probability threshold
            min_speech_duration_ms=250,    # Ignore speech shorter than 250ms
            min_silence_duration_ms=100    # Silence threshold to split chunks
        )

        for idx, ts in enumerate(speech_timestamps):
            start_sample = ts['start']
            end_sample = ts['end']

            # Slice the speech segment
            chunk_wav = wav[start_sample:end_sample]

            # Generate unique ID: <filename>_chunk_001, <filename>_chunk_002, etc.
            unique_chunk_id = f"{base_name}_chunk_{idx + 1:03d}"
            chunk_filename = f"{unique_chunk_id}.wav"
            chunk_path = os.path.join(OUTPUT_DIR, chunk_filename)

            # Save the isolated speech chunk
            save_audio(chunk_path, chunk_wav, sampling_rate=TARGET_SAMPLE_RATE)

            # Log chunk metadata
            chunks_manifest.append({
                "chunk_id": unique_chunk_id,
                "original_file": file_name,
                "chunk_path": chunk_path,
                "start_sec": round(start_sample / TARGET_SAMPLE_RATE, 3),
                "end_sec": round(end_sample / TARGET_SAMPLE_RATE, 3)
            })

            print(f"  Created chunk: {chunk_filename}")

    except Exception as e:
        print(f"Error running VAD on {file_name}: {e}")

# Save processing manifest
with open(MANIFEST_FILE, "w") as f:
    json.dump(chunks_manifest, f, indent=2)

print(f"\nVAD Chunking complete! Generated {len(chunks_manifest)} speech chunks.")
print(f"Manifest written to: {MANIFEST_FILE}")