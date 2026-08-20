import whisper
import torch
import torch.nn as nn

class WhisperIntentClassification(nn.Module):
    def __init__(self, model_type="small", n_class=20, dropout=0.3):
        super().__init__()
        self.encoder = whisper.load_model(model_type).encoder

        for param in self.encoder.parameters():
            param.requires_grad = True

        feature_dim = 768

        self.intent_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, n_class),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x, dim=1)
        intent = self.intent_classifier(x)
        return intent

