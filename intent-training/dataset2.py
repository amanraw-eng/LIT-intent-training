"""Adds duration-capped, length-masked samples on top of dataset.py's
HFIntentDataset, for use with model2.WhisperIntentClassification's optional
valid_lengths pooling arg.

Kept as a separate module (rather than changing dataset.py in place) so
trainer.py / train4.py, which import dataset.HFIntentDataset directly and
expect (mel, label) pairs, are unaffected.
"""
import torch
from torch.utils.data import Dataset
from whisper.audio import HOP_LENGTH, N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim

from dataset import _extract_audio_array

from librosa import resample as librosa_resample

# most clips are a few seconds long (median ~2.4s / p90 ~5.3s / p99 ~10.4s in
# data/intents10/bolAIndia_subset/data.jsonl) but every clip still gets
# zero-padded out to Whisper's fixed 30s window before the encoder (its
# positional embedding is a fixed 1500-timestep buffer - see
# whisper.model.AudioEncoder.forward's shape assert, so the encoder can't
# accept a shorter input tensor). Capping real content at 10s (covers ~99%
# of clips) and telling the model how many of the resulting 1500 encoder
# timesteps are real audio lets it pool over just those instead of
# averaging in the encoder's response to padding silence.
DURATION_CAP_S = 10.0
ENCODER_FRAMES_PER_SECOND = 50  # whisper encoder: 20ms/timestep
N_ENCODER_FRAMES = N_SAMPLES // HOP_LENGTH // 2  # 1500 for the standard 30s window


class HFIntentDataset(Dataset):
    """Same wrapping as dataset.HFIntentDataset, but each item also carries
    valid_len: the number of encoder timesteps (out of N_ENCODER_FRAMES)
    that are real audio rather than padding, after capping duration at
    DURATION_CAP_S."""

    def __init__(self, hf_dataset, intent_to_idx=None, duration_cap_s=DURATION_CAP_S):
        self.hf_dataset = hf_dataset
        if intent_to_idx is None:
            intents = sorted(set(hf_dataset["intent"]))
            intent_to_idx = {intent: idx for idx, intent in enumerate(intents)}
        self.intent_to_idx = intent_to_idx
        self.num_classes = len(self.intent_to_idx)
        self.max_samples = int(duration_cap_s * SAMPLE_RATE)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        audio, sr = _extract_audio_array(row["audio"])
        if sr != SAMPLE_RATE:
            audio = librosa_resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        real_samples = min(len(audio), self.max_samples)
        audio = audio[:real_samples]
        valid_len = min(
            N_ENCODER_FRAMES,
            max(1, round(real_samples / SAMPLE_RATE * ENCODER_FRAMES_PER_SECOND)),
        )

        audio = pad_or_trim(audio, N_SAMPLES)
        mel = log_mel_spectrogram(audio)
        label = self.intent_to_idx[row["intent"]]
        return mel, torch.tensor(label, dtype=torch.long), torch.tensor(valid_len, dtype=torch.long)


def collate_mel_fn(batch):
    mels, labels, valid_lengths = zip(*batch)
    return torch.stack(mels), torch.stack(labels), torch.stack(valid_lengths)
