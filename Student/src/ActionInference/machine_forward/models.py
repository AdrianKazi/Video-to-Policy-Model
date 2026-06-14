import torch
import torch.nn as nn


class MachineForwardModel(nn.Module):
    def __init__(self, z_dim=64, real_a_dim=2, hidden_dim=256, num_layers=1):
        super().__init__()

        self.history_encoder = nn.LSTM(
            input_size=z_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim + real_a_dim),
            nn.Linear(hidden_dim + real_a_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z_hist, a_real):
        _, (h_n, _) = self.history_encoder(z_hist)
        h_t = h_n[-1]
        x = torch.cat([h_t, a_real], dim=-1)
        return self.head(x)
