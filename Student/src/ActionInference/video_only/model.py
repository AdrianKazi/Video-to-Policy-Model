import torch
from torch import nn
from torch.nn import functional as F


class VideoDynamics(nn.Module):
    """Predict one candidate next frame per discrete latent action."""

    def __init__(self, history: int, codes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(history, 24, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 24, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(24, 16, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, codes, 3, padding=1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        residuals = self.decoder(self.encoder(history))
        return (history[:, -1:] + residuals).clamp(0.0, 1.0)


class LatentPolicy(nn.Module):
    """Choose a latent action from past and current frames."""

    def __init__(self, history: int, codes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(history, 16, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, codes),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.network(history)


def prediction_errors(
    predictions: torch.Tensor, current: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Motion-weighted squared error per example and candidate."""
    motion = F.max_pool2d((target - current).abs(), 5, stride=1, padding=2)
    weights = 1.0 + 12.0 * motion
    pixels = weights * (predictions - target).square()
    return pixels.sum(dim=(-2, -1)) / weights.sum(dim=(-2, -1))
