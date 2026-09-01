"""Real-noise extraction and telephony-style distortion for TTS augmentation.

Pulls quiet/background-noise segments out of REAL call recordings (not
synthetic/random noise) so augmented TTS audio picks up actual channel noise,
crosstalk murmur, and line artifacts from the same domain as the training data.
"""

import random

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, resample_poly


def extract_noise_donors(chunk_paths, max_donors=200, min_duration_s=2, scan_limit=3000, seed=0):
    """Scan real call-audio chunks, measure WHOLE-CLIP RMS, and take the
    quietest ones as noise donors.

    Earlier version took each chunk's quietest 15% of FRAMES - but nearly
    every chunk is itself a VAD-selected speech segment, so even its quietest
    frames are usually quiet speech/breath, not real ambient noise. Ranking by
    whole-clip RMS and keeping only entire clips that are quiet throughout is a
    much better proxy for genuine background/channel noise.

    Returns a list of (samples: float32 array, sample_rate).
    """
    rng = random.Random(seed)
    paths = list(chunk_paths)
    rng.shuffle(paths)
    paths = paths[:scan_limit]

    candidates = []
    for path in paths:
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) / sr < min_duration_s:
            continue
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        candidates.append((rms, audio, sr))

    if not candidates:
        raise RuntimeError("no usable noise donor candidates found in the given chunk_paths")

    candidates.sort(key=lambda c: c[0])
    return [(audio, sr) for _, audio, sr in candidates[:max_donors]]


def get_noise_segment(donors, n_samples, target_sr, rng=None):
    """Pick a random donor clip, resample to target_sr if needed, tile/crop to
    exactly n_samples, with a random start offset for variety."""
    rng = rng or random.Random()
    donor, sr = rng.choice(donors)
    if sr != target_sr:
        g = np.gcd(target_sr, sr)
        donor = resample_poly(donor, target_sr // g, sr // g)
    if len(donor) == 0:
        return np.zeros(n_samples, dtype=np.float32)
    reps = int(np.ceil(n_samples / len(donor))) + 1
    tiled = np.tile(donor, reps)
    start = rng.randint(0, max(0, len(tiled) - n_samples))
    return tiled[start : start + n_samples].astype(np.float32)


def mix_at_snr(clean, noise, snr_db):
    """Scale `noise` so the mix has the given signal-to-noise ratio, add it to
    `clean`. Both must be the same length."""
    clean_power = float(np.mean(clean.astype(np.float64) ** 2)) + 1e-12
    noise_power = float(np.mean(noise.astype(np.float64) ** 2)) + 1e-12
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return (clean + noise * scale).astype(np.float32)


def telephony_distort(audio, sr_in, sr_out=8000, band=(300, 3400)):
    """Bandpass to the telephony band and resample to sr_out, to make TTS
    audio match the source corpus's 8kHz call-recording character."""
    if sr_in != sr_out:
        g = np.gcd(sr_in, sr_out)
        audio = resample_poly(audio, sr_out // g, sr_in // g)
    nyq = sr_out / 2
    low = max(band[0] / nyq, 1e-3)
    high = min(band[1] / nyq, 1 - 1e-3)
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, audio)
    return filtered.astype(np.float32), sr_out


def augment(clean, sr_in, donors, snr_db_range=(8.0, 20.0), sr_out=8000, rng=None):
    """Full pipeline: bandpass+resample to telephony 8kHz, mix in a random real
    noise donor at a random SNR within snr_db_range. Returns (audio, sr_out)."""
    rng = rng or random.Random()
    distorted, sr = telephony_distort(clean, sr_in, sr_out=sr_out)
    noise = get_noise_segment(donors, len(distorted), sr, rng=rng)
    snr_db = rng.uniform(*snr_db_range)
    mixed = mix_at_snr(distorted, noise, snr_db)
    peak = np.max(np.abs(mixed)) + 1e-9
    if peak > 0.99:
        mixed = mixed / peak * 0.99
    return mixed.astype(np.float32), sr
