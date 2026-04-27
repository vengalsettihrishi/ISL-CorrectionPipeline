"""Two-head CTC model for gloss-guided structural regularization."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SharedBiLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 450,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        encoded, _ = self.lstm(x)
        return self.dropout(encoded)


class TwoHeadCTCModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        gloss_vocab_size: int,
        english_vocab_size: int,
        blank_id: int = 0,
    ):
        super().__init__()
        self.blank_id = blank_id
        self.encoder = SharedBiLSTMEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.gloss_head = nn.Linear(self.encoder.output_dim, gloss_vocab_size)
        self.english_head = nn.Linear(self.encoder.output_dim, english_vocab_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def gloss_log_probs(
        self,
        x: torch.Tensor,
        encoded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded = self.encoder(x) if encoded is None else encoded
        return torch.log_softmax(self.gloss_head(encoded), dim=-1)

    def english_log_probs(
        self,
        x: torch.Tensor,
        encoded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded = self.encoder(x) if encoded is None else encoded
        return torch.log_softmax(self.english_head(encoded), dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return self.gloss_log_probs(x, encoded), self.english_log_probs(x, encoded)


def freeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True

