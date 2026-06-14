"""Config and paths for the sequential latent model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from student_paths import student_dir


def _pkg() -> Path:
    return Path(__file__).resolve().parent


@dataclass
class SequentialPaths:
    package_root: Path = field(default_factory=_pkg)
    frames_dir: Path = field(init=False)
    runs_dir: Path = field(init=False)
    ae_runs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        stud = student_dir()
        self.frames_dir = stud / "data" / "frames"
        self.runs_dir = stud / "runs" / "sequential"
        self.ae_runs_dir = stud / "runs" / "autoencoder"

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.ae_runs_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class SequentialConfig:
    z_dim: int = 64
    seq_len: int = 32
    test_ratio: float = 0.2
    seed: int = 42
    batch_size: int = 32
    epochs: int = 40
    lr: float = 3e-4
    grad_clip: float = 1.0
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1

    # early stopping
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4

    device: Optional[str] = None

    paths: SequentialPaths = field(default_factory=SequentialPaths)
