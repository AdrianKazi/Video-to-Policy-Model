"""Build per-run AE train/test datasets from extracted frame folders."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from Autoencoder.config import AutoencoderConfig


def _load_episode(video_dir: Path) -> torch.Tensor:
    frames = sorted(video_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not frames:
        return torch.empty((0, 1, 84, 84), dtype=torch.float32)
    xs = [
        torch.tensor(np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0)
        for p in frames
    ]
    return torch.stack(xs).unsqueeze(1)


def _train_test_paths(run_dir: Path) -> tuple[Path, Path]:
    return run_dir / "ae_train_dataset.pt", run_dir / "ae_test_dataset.pt"


# def _new_run_dir(cfg: AutoencoderConfig) -> Path:
#     cfg.paths.runs_dir.mkdir(parents=True, exist_ok=True)
#     stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
#     run_dir = cfg.paths.runs_dir / f"autoencoder_{stamp}"
#     run_dir.mkdir(parents=True, exist_ok=True)
#     return run_dir

def _new_run_dir(cfg: AutoencoderConfig) -> Path:
    cfg.paths.runs_dir.mkdir(parents=True, exist_ok=True)
    return cfg.paths.runs_dir

def _build_sequences(input_dir: Path, episodes: list[Path]) -> list[torch.Tensor]:
    sequences = []
    for episode in episodes:
        if not episode.is_dir():
            continue
        seq_t = _load_episode(episode)
        if len(seq_t) > 1:
            sequences.append(seq_t)
            print(f"[dataset] episode {episode.name:>3} → len={len(seq_t)}")
    return sequences


def run_build_dataset(cfg: AutoencoderConfig, *, assume_yes: bool = False) -> Path:
    paths = cfg.paths
    frames = paths.frames_dir
    if not frames.is_dir() or not any(frames.iterdir()):
        raise RuntimeError(f"No frames under {frames}. Run `extract` first.")

    run_dir = _new_run_dir(cfg)
    out_train, out_test = _train_test_paths(run_dir)

    episodes = sorted((p for p in frames.iterdir() if p.is_dir()), key=lambda p: int(p.name))
    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)

    n_test = int(np.ceil(cfg.test_ratio * len(episodes)))
    test_idx = set(indices[:n_test].tolist())
    train_idx = set(indices[n_test:].tolist())

    train_eps = [episodes[i] for i in range(len(episodes)) if i in train_idx]
    test_eps = [episodes[i] for i in range(len(episodes)) if i in test_idx]

    if not train_eps or not test_eps:
        raise RuntimeError("Train/test split produced an empty side; adjust test_ratio or frame data.")

    print("\n[build] TRAIN episodes\n")
    train_sequences = _build_sequences(frames, train_eps)
    print("\n[build] TEST episodes\n")
    test_sequences = _build_sequences(frames, test_eps)

    torch.save(train_sequences, out_train)
    torch.save(test_sequences, out_test)
    print(f"\n[build] saved ae_train_dataset.pt ({len(train_sequences)} seq) → {out_train}")
    print(f"[build] saved ae_test_dataset.pt  ({len(test_sequences)} seq) → {out_test}")
    return run_dir
