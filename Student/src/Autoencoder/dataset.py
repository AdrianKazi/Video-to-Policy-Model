"""Torch datasets for episode tensors and flat frame batches."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class EpisodeTensorDataset(Dataset):
    """Each item is one episode tensor (T, 1, 84, 84), dtype float."""

    def __init__(self, pt_path: Path | str):
        self.samples: list[torch.Tensor] = torch.load(pt_path, map_location="cpu")
        if not isinstance(self.samples, list):
            raise TypeError("Expected list of episode tensors in .pt file")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


class FrameDataset(Dataset):
    """Flattens all episodes into independent frames (N, 1, 84, 84)."""

    def __init__(self, pt_path: Path | str):
        episodes = EpisodeTensorDataset(pt_path).samples
        frames: list[torch.Tensor] = []
        offsets: list[tuple[int, int]] = []
        for ep_i, seq in enumerate(episodes):
            t = seq.shape[0]
            for j in range(t):
                offsets.append((ep_i, j))
        self._episodes = episodes
        self._offsets = offsets

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ep_i, j = self._offsets[idx]
        return self._episodes[ep_i][j]
