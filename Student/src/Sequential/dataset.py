"""Sequential chunk datasets and dataloaders."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class SequentialChunkDataset(Dataset):
    def __init__(self, x: torch.Tensor, lengths: torch.Tensor, episode_ids: torch.Tensor):
        self.x = x
        self.lengths = lengths
        self.episode_ids = episode_ids

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[idx],
            "length": self.lengths[idx],
            "episode_id": self.episode_ids[idx],
        }


def build_dataloaders(
    train_data: dict[str, torch.Tensor],
    test_data: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = SequentialChunkDataset(
        train_data["x"],
        train_data["lengths"],
        train_data["episode_ids"],
    )
    test_ds = SequentialChunkDataset(
        test_data["x"],
        test_data["lengths"],
        test_data["episode_ids"],
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
