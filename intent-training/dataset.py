import io
import os
import random
from collections import defaultdict

# librosa lazy-loads its submodules on first attribute access (via
# lazy_loader) - `import librosa` alone does NOT pull in `resample`'s
# dependency chain. Importing the name directly forces that resolution now,
# in the main process, before DataLoader workers fork - otherwise the first
# access happens inside a forked worker and collides with wandb's import hook
# (wandb.sdk.lib.asyncio_manager.ForkedError).
from librosa import resample as librosa_resample
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from whisper.audio import N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim

DEFAULT_REPO_ID = "kapturecx/call-transcript-intent-data-v2"


def load_hf_split(repo_id, split, token=None):
    from datasets import load_dataset

    return load_dataset(repo_id, split=split, token=token or os.environ.get("HF_TOKEN"))


def _extract_audio_array(audio_field):
    if isinstance(audio_field, dict):
        array = audio_field.get("array")
        if array is not None:
            return np.asarray(array, dtype=np.float32), audio_field.get("sampling_rate", SAMPLE_RATE)
        raw_bytes = audio_field.get("bytes")
        if raw_bytes is not None:
            data, sr = sf.read(io.BytesIO(raw_bytes))
            return np.asarray(data, dtype=np.float32), sr

    # datasets>=5's Audio feature returns a lazy, torchcodec-backed AudioDecoder
    # instead of a plain dict - not a dict instance, but still supports
    # dict-style access for these two keys.
    try:
        return np.asarray(audio_field["array"], dtype=np.float32), audio_field["sampling_rate"]
    except (TypeError, KeyError):
        pass

    raise ValueError(f"Unsupported audio field: {type(audio_field)}")


class HFIntentDataset(Dataset):
    """Wraps a `datasets.Dataset` split pulled from the HF Hub (embedded audio
    bytes, native sample rate - currently 8kHz) into whisper-ready (mel, label)
    pairs. Pass the same `intent_to_idx` to every split built from one run so
    label indices stay consistent across train/validation/eval."""

    def __init__(self, hf_dataset, intent_to_idx=None):
        self.hf_dataset = hf_dataset
        if intent_to_idx is None:
            intents = sorted(set(hf_dataset["intent"]))
            intent_to_idx = {intent: idx for idx, intent in enumerate(intents)}
        self.intent_to_idx = intent_to_idx
        self.num_classes = len(self.intent_to_idx)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        audio, sr = _extract_audio_array(row["audio"])
        if sr != SAMPLE_RATE:
            audio = librosa_resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        audio = pad_or_trim(audio, N_SAMPLES)
        mel = log_mel_spectrogram(audio)
        label = self.intent_to_idx[row["intent"]]
        return mel, torch.tensor(label, dtype=torch.long)


def collate_mel_fn(batch):
    mels, labels = zip(*batch)
    return torch.stack(mels), torch.stack(labels)


def cap_dataset_per_class(hf_dataset, cap, seed=100, column="intent"):
    """Randomly subsample any class with more than `cap` rows down to `cap`
    (deterministic given `seed`); classes at or under `cap` are left
    untouched. Returns a new `datasets.Dataset` (row order sorted by original
    index, not shuffled)."""
    indices_by_intent = defaultdict(list)
    for i, intent in enumerate(hf_dataset[column]):
        indices_by_intent[intent].append(i)

    rng = random.Random(seed)
    capped_indices = []
    for indices in indices_by_intent.values():
        if len(indices) > cap:
            capped_indices.extend(rng.sample(indices, cap))
        else:
            capped_indices.extend(indices)
    capped_indices.sort()

    return hf_dataset.select(capped_indices)
