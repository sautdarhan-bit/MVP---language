"""A deliberately small sequence classifier: LSTM followed by a linear layer."""
import torch
from torch import nn
from common import FEATURES, FEATURE_VERSION


class SignLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(FEATURES, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, time, features]. Last hidden state summarizes the clip.
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])  # Raw logits for CrossEntropyLoss.


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["feature_version"] != FEATURE_VERSION:
        raise ValueError("Checkpoint uses incompatible features; extract and train again")
    model = SignLSTM(len(checkpoint["labels"]), checkpoint["hidden_size"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
