import os
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from demucs.pretrained import get_model
from demucs.apply import apply_model

OUTPUT_NOISE_DIR = '/mnt/HDD8TB/aman_ws/stt/LIT-intent-training/data/production_noise'
VALID_AUDIO_EXTENSIONS = ('.wav')

def extract_single_file(audio_path, output_dir=OUTPUT_NOISE_DIR, model=None):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Pretrained Demucs Model (if not already loaded)
    if model is None:
        model = get_model('htdemucs')
        model.eval()
        if torch.cuda.is_available():
            model.cuda()

    device = next(model.parameters()).device
    target_sr = model.samplerate

    # 2. Load and preprocess single audio file
    wav, sr = torchaudio.load(audio_path)
    orig_len = wav.shape[-1]

    # Resample to model sample rate if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)

    # Convert mono to stereo tensor expected by Demucs
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    # Add batch dimension: [1, Channels, Samples]
    wav_tensor = wav.unsqueeze(0).to(device)

    # 3. Perform Separation
    with torch.no_grad():
        sources = apply_model(model, wav_tensor)[0]  # Shape: [Stems, Channels, Samples]

    # Stems: 0: drums, 1: bass, 2: other, 3: vocals (speech)
    speech_stem = sources[3].cpu()
    noise_stem = (sources[0] + sources[1] + sources[2]).cpu()

    # 4. Crop back to match target sample length if resampled
    target_len = int(orig_len * (target_sr / sr))
    speech_stem = speech_stem[:, :target_len]
    noise_stem = noise_stem[:, :target_len]

    # 5. Downmix stereo back to mono for speech and noise
    speech_mono = speech_stem.mean(dim=0).numpy()
    noise_mono = noise_stem.mean(dim=0).numpy()

    # 6. Save outputs to disk
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    speech_path = os.path.join(output_dir, f"{base_name}_speech.wav")
    noise_path = os.path.join(output_dir, f"{base_name}_noise.wav")

    sf.write(speech_path, speech_mono, target_sr)
    sf.write(noise_path, noise_mono, target_sr)

    return speech_path, noise_path


def process_audio_directory(input_dir, output_dir=OUTPUT_NOISE_DIR):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Gather all supported audio files from input directory
    audio_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(VALID_AUDIO_EXTENSIONS)
    ]

    if not audio_paths:
        print(f"No audio files found in {input_dir}")
        return []

    # Load model once and reuse across iterations
    model = get_model('htdemucs')
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    extracted_files = []
    
    # Process files one by one with progress bar
    for path in tqdm(audio_paths, desc="Extracting audio stems"):
        try:
            sp, np_path = extract_single_file(path, output_dir=output_dir, model=model)
            extracted_files.append({"speech": sp, "noise": np_path})
        except Exception as e:
            print(f"\nFailed to process {os.path.basename(path)}: {e}")

    return extracted_files


# Example Usage:
INPUT_AUDIO_DIR = '/mnt/HDD8TB/aman_ws/stt/LIT-intent-training/data/August_data/user_utterances'
extracted_results = process_audio_directory(INPUT_AUDIO_DIR)