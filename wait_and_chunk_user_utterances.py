import json
import os
import shutil
import time
from pathlib import Path
import torch
from dotenv import load_dotenv

load_dotenv('.env')

# --- CONFIGURATION ---
INPUT_DIR = Path("/mnt/HDD8TB/aman_ws/stt/LIT-intent-training/data/August_data/user_utterances")
OUTPUT_DIR = Path("/mnt/HDD8TB/aman_ws/stt/LIT-intent-training/data/August_data/user_chunks")
PROCESSED_DIR = Path("/mnt/HDD8TB/aman_ws/stt/LIT-intent-training/data/August_data/processed_utterances")
MANIFEST_FILE = Path("/mnt/HDD8TB/aman_ws/stt/LIT-intent-training/data/August_data/user_chunks_manifest.jsonl")

FILE_THRESHOLD = 40000 
CHECK_INTERVAL_SEC = 30 * 60  # 30 minutes
TARGET_SAMPLE_RATE = 16000

# Safely create all output directories
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MANIFEST_FILE.parent, exist_ok=True)

# --- LOAD SILERO VAD MODEL ---
print("Loading Silero VAD model...")
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


def process_audio_files():
    """Runs VAD chunking on files currently inside INPUT_DIR."""
    valid_extensions = ('.wav', '.mp3', '.flac')
    audio_files = [f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting VAD processing on {len(audio_files)} files...")
    
    chunks_created = 0

    with open(MANIFEST_FILE, "a", encoding="utf-8") as manifest_f:
        for file_path in audio_files:
            base_name = file_path.stem

            try:
                wav = read_audio(str(file_path), sampling_rate=TARGET_SAMPLE_RATE)

                speech_timestamps = get_speech_timestamps(
                    wav,
                    model,
                    sampling_rate=TARGET_SAMPLE_RATE,
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100
                )

                for idx, ts in enumerate(speech_timestamps):
                    start_sample = ts['start']
                    end_sample = ts['end']
                    chunk_wav = wav[start_sample:end_sample]

                    unique_chunk_id = f"{base_name}_chunk_{idx + 1:03d}"
                    chunk_filename = f"{unique_chunk_id}.wav"
                    chunk_path = OUTPUT_DIR / chunk_filename

                    save_audio(str(chunk_path), chunk_wav, sampling_rate=TARGET_SAMPLE_RATE)

                    manifest_record = {
                        "chunk_id": unique_chunk_id,
                        "original_file": file_path.name,
                        "chunk_path": str(chunk_path),
                        "start_sec": round(start_sample / TARGET_SAMPLE_RATE, 3),
                        "end_sec": round(end_sample / TARGET_SAMPLE_RATE, 3),
                        "processed_at": time.strftime('%Y-%m-%dT%H:%M:%SZ')
                    }
                    manifest_f.write(json.dumps(manifest_record) + "\n")
                    chunks_created += 1

                # Safely relocate original file to prevent duplicate processing
                shutil.move(str(file_path), str(PROCESSED_DIR / file_path.name))

            except Exception as e:
                print(f"Error running VAD on {file_path.name}: {e}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing complete. Created {chunks_created} chunks.")


def main():
    print(f"Watcher started. Monitoring '{INPUT_DIR}' every 30 minutes for > {FILE_THRESHOLD} files...")
    
    while True:
        try:
            # Ensure folder existence check before reading contents
            os.makedirs(INPUT_DIR, exist_ok=True)
            current_count = sum(1 for item in INPUT_DIR.iterdir() if item.is_file())
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Current file count: {current_count}/{FILE_THRESHOLD}")

            if current_count > FILE_THRESHOLD:
                print("Threshold reached! Triggering VAD chunking workflow...")
                process_audio_files()
            else:
                print("Threshold not met. Sleeping for 30 minutes...")

        except Exception as e:
            print(f"Unexpected error in watcher loop: {e}")

        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()