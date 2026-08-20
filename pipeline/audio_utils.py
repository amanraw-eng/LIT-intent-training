import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_pcm16(path, target_sr=16000):
    """Load a wav file, downmix to mono, resample to target_sr, return (pcm16_bytes, duration_s)."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        g = np.gcd(target_sr, sr)
        data = resample_poly(data, target_sr // g, sr // g)
    data = np.clip(data, -1.0, 1.0)
    pcm16 = (data * 32767.0).astype(np.int16)
    duration_s = len(data) / target_sr
    return pcm16.tobytes(), duration_s
