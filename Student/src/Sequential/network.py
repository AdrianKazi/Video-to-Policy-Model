"""Sequential latent predictor."""
from __future__ import annotations

import torch
import torch.nn as nn


class LatentLSTM(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=z_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z_seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                z_seq,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(z_seq)
        return self.head(h_n[-1])
