import torch
import torch.nn as nn


def overlay_frames(x_seq, decay=0.9):
    # x_seq: (B,T,1,84,84)
    T = x_seq.shape[1]
    weights = torch.tensor(
        [decay ** (T - 1 - i) for i in range(T)],
        device=x_seq.device,
        dtype=x_seq.dtype,
    )
    weights = weights / weights.sum()
    return (x_seq * weights.view(1, T, 1, 1, 1)).sum(dim=1)  # (B,1,84,84)


class IDMModel(nn.Module):
    def __init__(self, c_dim, latent_a_dim=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(c_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, latent_a_dim),
            nn.Tanh(),
        )

    def forward(self, c_t):
        return self.net(c_t)


class ActionAdapter(nn.Module):
    def __init__(self, latent_a_dim=8, real_a_dim=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_a_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, real_a_dim),
            nn.Tanh(),
        )

    def forward(self, latent_a):
        return self.net(latent_a)


class FDMModel(nn.Module):
    def __init__(self, c_dim, z_dim=64, latent_a_dim=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(c_dim + latent_a_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, z_dim),
        )

    def forward(self, c_t, latent_a):
        x = torch.cat([c_t, latent_a], dim=-1)
        return self.net(x)
