"""Hyperparameters and paths for the Autoencoder module."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from student_paths import repo_root, student_dir


def _pkg() -> Path:
    return Path(__file__).resolve().parent


@dataclass
class AutoencoderPaths:
    """Student filesystem layout for the Autoencoder pipeline."""

    package_root: Path = field(default_factory=_pkg)
    frames_dir: Path = field(init=False)
    runs_dir: Path = field(init=False)
    teacher_videos: Path = field(init=False)

    def __post_init__(self) -> None:
        stud = student_dir()
        # Raw extracted frames (shared with teacher prep script)
        self.frames_dir = stud / "data" / "frames"

        # Self-contained AE runs live under Student/runs/autoencoder/
        self.runs_dir = stud / "runs" / "autoencoder"
        self.teacher_videos = repo_root() / "Teacher" / "videos"

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AutoencoderConfig:
    """Training and data knobs (edit here or override via CLI)."""

    z_dim: int = 64
    lr: float = 3e-4
    epochs: int = 40
    batch_size: int = 64
    grad_clip: float = 1.0
    num_workers: int = 0
    log_every: int = 50
    # If set, train on a random subset of frames (faster notebook / smoke tests; None = full data)
    limit_train_samples: Optional[int] = None
    # weighted_mse
    wmse_threshold: float = 0.1
    wmse_high_weight: float = 20.0
    # build_dataset
    test_ratio: float = 0.2
    seed: int = 42
        # early stopping
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4

    # eval
    recon_rows: int = 20
    device: Optional[str] = None  # None → auto (cuda > mps > cpu)

    paths: AutoencoderPaths = field(default_factory=AutoencoderPaths)
