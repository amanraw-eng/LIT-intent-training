import os
import json
import torch
import torchaudio

MANIFEST_FILE = "/mnt/HDD8TB/aman_ws/stt/data/user_example_chunks_manifest.json"
CLEANED_MANIFEST_FILE = "/mnt/HDD8TB/aman_ws/stt/data/chunks_manifest_cleaned.json"

# Adjust thresholds based on your audio conditions:
MIN_RMS_DB = -38.0   # Chunks quieter than -38 dBFS are considered background noise/silence
MAX_ZCR = 0.25       # High zero-crossing rate (>0.25) with low energy indicates static/hiss

def calculate_rms_db(waveform: torch.Tensor) -> float:
    """Calculates Root Mean Square (RMS) energy in dBFS."""
    rms = torch.sqrt(torch.mean(waveform ** 2))
    db = 20 * torch.log10(rms + 1e-9)
    return db.item()

def calculate_zcr(waveform: torch.Tensor) -> float:
    """Calculates Zero-Crossing Rate (ZCR) to detect unvoiced static noise."""
    zero_crossings = torch.sum(torch.abs(torch.diff(torch.sign(waveform)))) / 2
    return (zero_crossings / waveform.numel()).item()

# Load manifest
if os.path.exists(MANIFEST_FILE):
    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
else:
    raise FileNotFoundError(f"Could not find {MANIFEST_FILE}. Run Step 2 first.")

cleaned_manifest = []
deleted_count = 0

print("Scanning chunks for noise...\n")

for item in manifest:
    chunk_path = item["chunk_path"]
    
    if not os.path.exists(chunk_path):
        continue

    waveform, sample_rate = torchaudio.load(chunk_path)
    
    rms_db = calculate_rms_db(waveform)
    zcr = calculate_zcr(waveform)

    # Condition 1: Pure quiet/ambient noise
    is_too_quiet = rms_db < MIN_RMS_DB
    
    # Condition 2: High static/hiss without significant speech energy
    is_static_hiss = zcr > MAX_ZCR and rms_db < (MIN_RMS_DB + 8)

    if is_too_quiet or is_static_hiss:
        reason = "Too Quiet" if is_too_quiet else "Static Noise"
        print(f"❌ Deleting [{reason}]: {os.path.basename(chunk_path)} (RMS: {rms_db:.1f} dB | ZCR: {zcr:.2f})")
        
        # Delete the noise file
        os.remove(chunk_path)
        deleted_count += 1
    else:
        cleaned_manifest.append(item)

# Save updated manifest
with open(CLEANED_MANIFEST_FILE, "w") as f:
    json.dump(cleaned_manifest, f, indent=2)

print("\n--- Summary ---")
print(f"Total processed: {len(manifest)}")
print(f"Noise chunks deleted: {deleted_count}")
print(f"Valid speech chunks remaining: {len(cleaned_manifest)}")
print(f"Updated manifest saved to: {CLEANED_MANIFEST_FILE}")
