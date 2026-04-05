"""
model.py — Bidirectional LSTM classifier for ISL landmark sequences.

Architecture:
    Input: (batch, seq_length, features)  e.g. (32, 30, 225)
      → Layer Normalization (stabilizes landmark inputs)
      → Bidirectional LSTM (2 layers, 128 hidden)
      → Temporal attention pooling (learns which frames matter most)
      → Dropout
      → Fully connected → num_classes

Why this architecture:
    - LayerNorm: Landmark coordinates have varying scales (x,y in [0,1],
      z can be negative). Normalization helps training stability.
    - Bidirectional LSTM: Signs have temporal structure in both directions.
      The ending hand position is as informative as the starting position.
    - Attention pooling: Not all frames are equally important. The middle
      of a sign (peak of the gesture) carries more information than the
      transition frames at start/end. Attention learns this automatically.
    - The model is small (~2-4M parameters) and runs in <50ms on a Pi 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class TemporalAttention(nn.Module):
    """
    Learnable attention over the temporal axis.

    Instead of just taking the last LSTM hidden state (which loses
    information from early frames) or mean-pooling (which weights all
    frames equally), attention learns which frames matter most for
    each class.

    Output: weighted sum of all LSTM outputs.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            (batch, hidden_size) — attention-weighted representation
        """
        # (batch, seq_len, 1)
        scores = self.attention(lstm_output)
        weights = F.softmax(scores, dim=1)

        # Weighted sum: (batch, hidden_size)
        context = (lstm_output * weights).sum(dim=1)
        return context


class ISLClassifier(nn.Module):
    """
    Full classifier: LayerNorm → BiLSTM → Attention → FC → Classes
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Normalize input features
        self.layer_norm = nn.LayerNorm(config.input_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )

        # Effective hidden size doubles if bidirectional
        effective_hidden = config.hidden_size * (2 if config.bidirectional else 1)

        # Attention pooling
        self.attention = TemporalAttention(effective_hidden)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(effective_hidden, effective_hidden // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout / 2),
            nn.Linear(effective_hidden // 2, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_length, features) — landmark sequences
        Returns:
            (batch, num_classes) — logits (not softmax)
        """
        # Normalize
        x = self.layer_norm(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention pooling
        context = self.attention(lstm_out)  # (batch, hidden*2)

        # Classify
        logits = self.classifier(context)  # (batch, num_classes)

        return logits

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference.
        Returns class indices, not logits.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_with_confidence(self, x: torch.Tensor) -> tuple:
        """
        Returns (predicted_class, confidence_score) for each sample.
        Useful for the smoothing layer downstream.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            return predicted, confidence


def build_model(config: ModelConfig) -> ISLClassifier:
    """Factory function to create a model from config."""
    model = ISLClassifier(config)
    print(f"Model created: {model.count_parameters():,} trainable parameters")
    return model
