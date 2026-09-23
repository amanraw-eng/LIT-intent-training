"""v2 architecture: deeper classification head + masked-mean pooling over
just the real (non-padding) audio frames. Ported as-is from
intent-training/model2.py (expt/all branch) - see intent_service_v2.py for
why callers MUST pass valid_lengths to forward() for this class, unlike v1's
WhisperIntentClassification (services/whisper_model.py)."""

import whisper
import torch
import torch.nn as nn


class WhisperIntentClassification(nn.Module):
    def __init__(self, model_type="small", n_class=15, dropout=0.2):
        super().__init__()
        self.encoder = whisper.load_model(model_type).encoder

        for param in self.encoder.parameters():
            param.requires_grad = True

        feature_dim = self.encoder.ln_post.normalized_shape[0]

        self.intent_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, n_class),
        )

    def forward(self, x, valid_lengths=None):
        x = self.encoder(x)  # Shape: [B, T, D]

        if valid_lengths is not None:
            t = x.shape[1]
            positions = torch.arange(t, device=x.device).unsqueeze(0)  # [1, T]
            mask = (positions < valid_lengths.unsqueeze(1)).unsqueeze(-1).to(x.dtype)  # [B, T, 1]
            counts = valid_lengths.clamp(min=1).to(x.dtype).unsqueeze(-1)  # [B, 1]
            x = (x * mask).sum(dim=1) / counts
        else:
            x = torch.mean(x, dim=1)

        intent = self.intent_classifier(x)
        return intent
