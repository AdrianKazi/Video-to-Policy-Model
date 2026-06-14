import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """84×84 grayscale conv autoencoder."""

    def __init__(self, z_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(25600, z_dim),
            nn.LayerNorm(z_dim),
        )

        self.decoder_fc = nn.Linear(z_dim, 25600)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder_fc(z)
        x_hat = x_hat.view(-1, 256, 10, 10)
        x_hat = self.decoder(x_hat)
        x_hat = x_hat[:, :, :84, :84]
        return x_hat, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder_fc(z)
        x_hat = x_hat.view(-1, 256, 10, 10)
        x_hat = self.decoder(x_hat)
        return x_hat[:, :, :84, :84]
