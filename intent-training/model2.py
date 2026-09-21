import whisper
import torch
import torch.nn as nn

# Same layer shapes as model.py's WhisperIntentClassification, so a checkpoint
# trained with either module loads into the other via strict state_dict
# loading (api.py, push_to_hub.py). The only change is forward() optionally
# doing length-masked mean pooling instead of averaging over all 1500 encoder
# timesteps - most clips run a few seconds inside a 30s-padded window, so an
# unmasked mean is dominated by the encoder's response to padding silence.
ENCODER_FRAMES_PER_SECOND = 50  # whisper encoder output: 20ms/timestep


class WhisperIntentClassification(nn.Module):
    def __init__(self, model_type="small", n_class=20, dropout=0.3 ):
        super().__init__()
        self.encoder = whisper.load_model(model_type).encoder

        for param in self.encoder.parameters():
            param.requires_grad = True

        feature_dim = 768

        self.intent_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(dropout),
            nn.Linear(128, n_class)
        )

    def forward(self, x, valid_lengths=None):
        x = self.encoder(x)  # [B, T=1500, D]

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
