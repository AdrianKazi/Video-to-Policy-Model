"""Build sequential predataset and chunk datasets inside a run folder."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from Sequential.config import SequentialConfig


# def _new_run_dir(cfg: SequentialConfig) -> Path:
#     cfg.paths.runs_dir.mkdir(parents=True, exist_ok=True)
#     stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
#     run_dir = cfg.paths.runs_dir / f"sequential_{stamp}"
#     run_dir.mkdir(parents=True, exist_ok=True)
#     return run_dir

def _new_run_dir(cfg: SequentialConfig) -> Path:
    cfg.paths.runs_dir.mkdir(parents=True, exist_ok=True)
    return cfg.paths.runs_dir


def load_episode(video_dir: Path) -> np.ndarray:
    frames = sorted(video_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not frames:
        return np.empty((0, 84, 84), dtype=np.float32)
    xs = [np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0 for p in frames]
    return np.stack(xs)


def sim_func(prev: np.ndarray, curr: np.ndarray, threshold: float = 0.01) -> bool:
    prev_v = prev.reshape(-1).astype(np.float32)
    curr_v = curr.reshape(-1).astype(np.float32)
    alpha = np.dot(curr_v, prev_v) / (np.dot(prev_v, prev_v) + 1e-8)
    proj = alpha * prev_v
    residual = curr_v - proj
    novelty_ratio = np.linalg.norm(residual) / (np.linalg.norm(curr_v) + 1e-8)
    return novelty_ratio > threshold


def run_build_dataset(cfg: SequentialConfig) -> Path:
    frames_root = cfg.paths.frames_dir
    if not frames_root.is_dir() or not any(frames_root.iterdir()):
        raise RuntimeError(f"No frames under {frames_root}. Build/extract them first.")

    run_dir = _new_run_dir(cfg)
    predataset_pt = run_dir / "predataset.pt"
    seq_train_pt = run_dir / "seq_train_dataset.pt"
    seq_test_pt = run_dir / "seq_test_dataset.pt"

    video_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()], key=lambda p: int(p.name))

    predataset: list[torch.Tensor] = []
    episode_lengths_raw: list[int] = []
    episode_lengths_reduced: list[int] = []

    for video_dir in video_dirs:
        x = load_episode(video_dir)
        t = len(x)
        episode_lengths_raw.append(t)

        if t == 0:
            predataset.append(torch.empty((0, 1, 84, 84), dtype=torch.float32))
            episode_lengths_reduced.append(0)
            continue

        curr_frame = x[0]
        frame_seq = [curr_frame]
        for idx in range(1, t):
            next_frame = x[idx]
            if sim_func(curr_frame, next_frame):
                curr_frame = next_frame
                frame_seq.append(curr_frame)

        predataset.append(torch.tensor(np.stack(frame_seq), dtype=torch.float32).unsqueeze(1))
        episode_lengths_reduced.append(len(frame_seq))

    predataset_obj = {
        "episodes": predataset,
        "episode_lengths_raw": torch.tensor(episode_lengths_raw, dtype=torch.int32),
        "episode_lengths_reduced": torch.tensor(episode_lengths_reduced, dtype=torch.int32),
    }
    torch.save(predataset_obj, predataset_pt)

    episodes = predataset_obj["episodes"]
    chunks: list[torch.Tensor] = []
    chunk_lengths: list[int] = []
    episode_ids: list[int] = []

    for ep_idx, ep in enumerate(episodes):
        t = int(ep.shape[0])
        if t == 0:
            continue
        for start in range(0, t, cfg.seq_len):
            chunk = ep[start : start + cfg.seq_len]
            valid_len = int(chunk.shape[0])
            if valid_len < cfg.seq_len:
                pad = torch.zeros((cfg.seq_len - valid_len, 1, 84, 84), dtype=chunk.dtype)
                chunk = torch.cat([chunk, pad], dim=0)
            chunks.append(chunk)
            chunk_lengths.append(valid_len)
            episode_ids.append(ep_idx)

    if not chunks:
        raise RuntimeError("No sequential chunks were built from the predataset.")

    x = torch.stack(chunks)
    lengths = torch.tensor(chunk_lengths, dtype=torch.int32)
    ep_ids = torch.tensor(episode_ids, dtype=torch.int32)

    all_episode_ids = np.arange(len(episodes))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(all_episode_ids)
    test_size = int(np.ceil(len(episodes) * cfg.test_ratio))
    test_episode_ids = set(all_episode_ids[:test_size].tolist())
    train_episode_ids = set(all_episode_ids[test_size:].tolist())

    train_mask = torch.tensor([int(ep.item()) in train_episode_ids for ep in ep_ids], dtype=torch.bool)
    test_mask = torch.tensor([int(ep.item()) in test_episode_ids for ep in ep_ids], dtype=torch.bool)

    if not bool(train_mask.any()) or not bool(test_mask.any()):
        raise RuntimeError("Sequential train/test split produced an empty side; adjust test_ratio or data.")

    seq_train_data = {
        "x": x[train_mask],
        "lengths": lengths[train_mask],
        "episode_ids": ep_ids[train_mask],
        "seq_len": cfg.seq_len,
    }
    seq_test_data = {
        "x": x[test_mask],
        "lengths": lengths[test_mask],
        "episode_ids": ep_ids[test_mask],
        "seq_len": cfg.seq_len,
    }

    torch.save(seq_train_data, seq_train_pt)
    torch.save(seq_test_data, seq_test_pt)
    print(f"[dataset][seq] saved → {run_dir}")
    return run_dir
