import json
import os
from pathlib import Path
import requests
import torchaudio

# Input configuration
JSON_FILE = "xPertVoiceAug26PQcallrec.json"
OUTPUT_DIR = "user_utterances"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(JSON_FILE, "r") as f:
    recordings = json.load(f)

for item in recordings:
    url = item["recording_url"]
    filename = Path(url).name
    temp_file = f"temp_{filename}"

    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save downloaded audio temporarily
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Load audio (waveform shape: [channels, num_samples])
        waveform, sample_rate = torchaudio.load(temp_file)

        if waveform.shape[0] < 2:
            print(f"Warning: {filename} does not have a 2nd channel (channels found: {waveform.shape[0]}). Skipping...")
        else:
            # Channel 0 = Bot, Channel 1 = User
            user_channel = waveform[1:2, :]  # Retain 2D shape [1, samples]

            output_path = os.path.join(OUTPUT_DIR, filename)
            torchaudio.save(output_path, user_channel, sample_rate)
            print(f"Saved user audio to: {output_path}")

    except Exception as e:
        print(f"Error processing {url}: {e}")

    finally:
        # Clean up temporary download file
        if os.path.exists(temp_file):
            os.remove(temp_file)

print("\nChannel extraction complete!")