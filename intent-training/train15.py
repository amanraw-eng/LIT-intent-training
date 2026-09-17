import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import mlflow
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, Dataset
from whisper.audio import HOP_LENGTH, N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim

from dataset_paths import resolve_chunk_path
from experiment_config import EXPERIMENT_VERSION_ENV, PROJECT_ROOT, get_experiment_paths, section
from model2 import WhisperIntentClassification

# ============================================================
# COMMON SETTINGS (edit shared_settings.py)
# ============================================================
_TRAINING = section("training")
_AUGMENTED17 = _TRAINING["augmented17"]
SEED = _TRAINING["seed"]
GPU_DEVICE = _TRAINING["gpu_device"]
DATA_DIR = str(PROJECT_ROOT / section("project")["data_dir"])
JSONL_DATA_PATH = str(PROJECT_ROOT / _AUGMENTED17["jsonl_data_path"])
DEFAULT_AUDIO_DIR = str(PROJECT_ROOT / "data/augmented_data/audio")
WHISPER_SIZE = _TRAINING["whisper_size"]
WEIGHT_DECAY = _TRAINING["weight_decay"]

# ============================================================
# SCRIPT-SPECIFIC RUN OVERRIDES (edit here for this run)
# ============================================================
BATCH_SIZE = 16
NUM_WORKERS = 4
MAX_EPOCHS = 15
PATIENCE = 5
PRECISION = "16-mixed"
ACCUMULATE_GRAD_BATCHES = 2

# most clips are a few seconds long (median ~2.4s / p90 ~5.3s / p99 ~10.4s in
# data/intents10/bolAIndia_subset/data.jsonl) but every clip still gets
# zero-padded to Whisper's fixed 30s window before the encoder, so an
# unmasked mean-pool over all 1500 encoder timesteps is mostly averaging in
# the encoder's response to silence. Capping real content at 10s (covers
# ~99% of clips) and pooling only over the corresponding encoder timesteps
# (via model2.WhisperIntentClassification's valid_lengths arg) keeps the
# signal from being diluted, without touching the encoder's input shape
# (still padded to 30s - required by its fixed positional embedding buffer).
DURATION_CAP_S = 10.0
ENCODER_FRAMES_PER_SECOND = 50  # whisper encoder: 20ms/timestep
N_ENCODER_FRAMES = N_SAMPLES // HOP_LENGTH // 2  # 1500 for the standard 30s window

# training accuracy climbing while val loss rose from the first eval onward
# (both the batch-8 and batch-16 runs) pointed at the whole encoder being
# fine-tuned at the same LR as a randomly-initialized head from step 0 -
# the head's large early gradients corrupt pretrained acoustic weights
# before the head has learned anything. Freeze the encoder while the head
# warms up, then unfreeze it at a much lower LR than the head.
FREEZE_ENCODER_EPOCHS = 3
ENCODER_LR = 1e-6
HEAD_LR = 5e-4

# Reproducibility
pl.seed_everything(SEED)
torch.manual_seed(SEED)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE

# ============================================================
# LOCAL JSONL DATASET & COLLATOR
# ============================================================
class LocalAudioIntentDataset(Dataset):
    def __init__(self, records, intent_to_idx, audio_dir, duration_cap_s=DURATION_CAP_S):
        self.records = records
        self.intent_to_idx = intent_to_idx
        self.audio_dir = audio_dir
        self.max_samples = int(duration_cap_s * SAMPLE_RATE)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        audio_path = resolve_chunk_path(item["chunk_path"], self.audio_dir)
        intent_str = item["intent"]
        label_idx = self.intent_to_idx[intent_str]

        # Load audio, convert to 16kHz mono, extract Mel-Spectrogram
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)

        # cap real content at DURATION_CAP_S; still zero-pad to Whisper's
        # required 30s window below (its positional embedding is a fixed
        # 1500-timestep buffer, see whisper.model.AudioEncoder.forward's
        # shape assert - the encoder can't accept a shorter tensor), but
        # record how many encoder timesteps are real audio so the model can
        # pool over just those instead of averaging in padding silence.
        real_samples = min(waveform.shape[0], self.max_samples)
        waveform = waveform[:real_samples]
        valid_len = min(
            N_ENCODER_FRAMES,
            max(1, round(real_samples / SAMPLE_RATE * ENCODER_FRAMES_PER_SECOND)),
        )

        audio_padded = pad_or_trim(waveform, N_SAMPLES)
        mel = log_mel_spectrogram(audio_padded)

        return mel, torch.tensor(label_idx, dtype=torch.long), torch.tensor(valid_len, dtype=torch.long)

def collate_mel_fn(batch):
    mels = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    valid_lengths = torch.stack([item[2] for item in batch])
    return mels, labels, valid_lengths

# ============================================================
# PYTORCH LIGHTNING MODULE
# ============================================================
class LightningModel(pl.LightningModule):
    def __init__(
        self,
        n_class,
        whisper_size=WHISPER_SIZE,
        encoder_lr=ENCODER_LR,
        head_lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY,
        freeze_encoder_epochs=FREEZE_ENCODER_EPOCHS,
        class_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.whisper_size = whisper_size
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.freeze_encoder_epochs = freeze_encoder_epochs

        self.model = WhisperIntentClassification(whisper_size, n_class=n_class)
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(n_class),
        )

    def forward(self, x, valid_lengths=None):
        return self.model(x, valid_lengths=valid_lengths)

    def on_train_epoch_start(self):
        freeze = self.current_epoch < self.freeze_encoder_epochs
        for p in self.model.encoder.parameters():
            p.requires_grad = not freeze
        if self.current_epoch == self.freeze_encoder_epochs:
            print(f"[train15] epoch {self.current_epoch}: unfreezing encoder (lr={self.encoder_lr})")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.encoder.parameters(), "lr": self.encoder_lr},
                {"params": self.model.intent_classifier.parameters(), "lr": self.head_lr},
            ],
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/acc",
            },
        }

    def loss_fn(self, prediction, targets):
        return nn.CrossEntropyLoss(weight=self.class_weights)(prediction, targets)

    def training_step(self, batch, batch_idx):
        x, y, valid_lengths = batch
        y = y.view(-1)
        logits = self(x, valid_lengths=valid_lengths)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        acc = (winners == y).sum().float() / float(logits.size(0))

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        x, y, valid_lengths = batch
        y = y.view(-1)

        logits = self(x, valid_lengths=valid_lengths)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        acc = (winners == y).sum().float() / float(logits.size(0))

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-version",
        help=f"Experiment version (defaults to {EXPERIMENT_VERSION_ENV} or v5)",
    )
    parser.add_argument(
        "--audio-dir", default=DEFAULT_AUDIO_DIR,
        help="Folder containing audio files referenced by relative chunk_path values",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path to pre-trained PyTorch Lightning checkpoint (.ckpt) to fine-tune from",
    )
    args = parser.parse_args()

    # Setup paths and MLFlow
    experiment_paths = get_experiment_paths(args.experiment_version)
    print(f"Experiment version: {experiment_paths.version}")
    print(f"Checkpoint directory: {experiment_paths.checkpoint_dir}")
    print(f"Intent map output: {experiment_paths.intent_map_path}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set in .env file.")
    mlflow.set_tracking_uri(tracking_uri)

    if token := os.environ.get("MLFLOW_TRACKING_TOKEN"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    experiment = mlflow.set_experiment(experiment_paths.mlflow_experiment_name)
    print(f"MLFlow Experiment '{experiment_paths.mlflow_experiment_name}' -> ID={experiment.experiment_id}")

    # Load and validate local dataset
    if not os.path.exists(JSONL_DATA_PATH):
        raise FileNotFoundError(f"Dataset JSONL file not found at: {JSONL_DATA_PATH}")

    valid_records = []
    with open(JSONL_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            if resolve_chunk_path(rec.get("chunk_path", ""), args.audio_dir).exists():
                valid_records.append(rec)

    print(f"Loaded {len(valid_records)} valid records with existing audio files.")

    # Train / Validation Split (85% / 15%)
    torch.manual_seed(SEED)
    indices = torch.randperm(len(valid_records)).tolist()
    split_idx = int(0.85 * len(valid_records))
    train_records = [valid_records[i] for i in indices[:split_idx]]
    val_records = [valid_records[i] for i in indices[split_idx:]]

    # Map Intents
    intents = sorted(list({r["intent"] for r in valid_records}))
    intent_to_idx = {intent: idx for idx, intent in enumerate(intents)}
    n_class = len(intent_to_idx)
    print(f"Found {n_class} distinct intent classes.")

    os.makedirs(os.path.dirname(experiment_paths.intent_map_path), exist_ok=True)
    with open(experiment_paths.intent_map_path, "w", encoding="utf-8") as f:
        json.dump(intent_to_idx, f, indent=2, ensure_ascii=False)

    # Class Weights setup
    train_labels = [intent_to_idx[r["intent"]] for r in train_records]
    class_counts = Counter(train_labels)
    class_weights = torch.tensor(
        [len(train_labels) / (n_class * class_counts[i]) for i in range(n_class)],
        dtype=torch.float32,
    )

    # DataLoaders
    train_dataset = LocalAudioIntentDataset(train_records, intent_to_idx, args.audio_dir)
    val_dataset = LocalAudioIntentDataset(val_records, intent_to_idx, args.audio_dir)

    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_mel_fn,
        pin_memory=True,
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_mel_fn,
        pin_memory=True,
    )

    # Model Setup (Checkpoint loading or Initializing)
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        print(f"--> Loading pre-trained checkpoint from: {args.ckpt_path}")
        model = LightningModel.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            n_class=n_class,
            whisper_size=WHISPER_SIZE,
            encoder_lr=ENCODER_LR,
            head_lr=HEAD_LR,
            weight_decay=WEIGHT_DECAY,
            freeze_encoder_epochs=FREEZE_ENCODER_EPOCHS,
            class_weights=class_weights,
        )
    else:
        print("--> Initializing model weights from scratch.")
        model = LightningModel(
            n_class=n_class,
            whisper_size=WHISPER_SIZE,
            encoder_lr=ENCODER_LR,
            head_lr=HEAD_LR,
            weight_decay=WEIGHT_DECAY,
            freeze_encoder_epochs=FREEZE_ENCODER_EPOCHS,
            class_weights=class_weights,
        )

    # Callbacks and Logger
    logger = MLFlowLogger(
        experiment_name=experiment_paths.mlflow_experiment_name,
        run_name=experiment_paths.run_name,
        tracking_uri=tracking_uri,
    )

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_paths.checkpoint_dir,
        monitor='val/acc',
        mode='max',
        save_top_k=4,
        verbose=1,
        filename=experiment_paths.checkpoint_prefix + "-epoch{epoch:02d}-val_acc{val/acc:.3f}"
    )

    early_stopping_callback = EarlyStopping(
        monitor='val/acc',
        mode='max',
        patience=PATIENCE,
        verbose=True
    )

    # Trainer setup
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        precision=PRECISION,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        enable_checkpointing=True,
        callbacks=[
            model_checkpoint_callback,
            early_stopping_callback,
        ],
        logger=logger,
    )

    # Fit Model
    trainer.fit(
        model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
        ckpt_path=args.ckpt_path if (args.ckpt_path and os.path.exists(args.ckpt_path)) else None
    )
