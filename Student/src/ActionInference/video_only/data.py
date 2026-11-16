from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


@dataclass
class Episode:
    path: Path
    frames: np.ndarray


def read_videos(directory: Path, size: int, minimum_frames: int, seed: int, limit):
    paths = sorted(path for path in directory.glob("*.mp4") if path.is_file())
    if limit is not None:
        order = np.random.default_rng(seed).permutation(len(paths))[:limit]
        paths = [paths[index] for index in sorted(order)]
    episodes = []
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        frames = []
        try:
            if not capture.isOpened():
                raise OSError(f"Could not open {path}")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA))
        finally:
            capture.release()
        if len(frames) >= minimum_frames:
            episodes.append(Episode(path, np.stack(frames)))
    if len(episodes) < 2:
        raise ValueError(
            f"Need at least two usable MP4 episodes with {minimum_frames} frames in {directory}"
        )
    return episodes


def split_episodes(episodes: list[Episode], seed: int):
    order = np.random.default_rng(seed).permutation(len(episodes))
    held_out = max(1, round(len(episodes) * 0.2))
    train = [episodes[index] for index in order[held_out:]]
    test = [episodes[index] for index in order[:held_out]]
    return train, test


class VideoWindows(Dataset):
    def __init__(self, episodes: list[Episode], history: int, horizon: int, stride: int = 1):
        self.episodes = episodes
        self.history = history
        self.horizon = horizon
        self.windows = [
            (episode_index, start)
            for episode_index, episode in enumerate(episodes)
            for start in range(0, len(episode.frames) - history - horizon + 1, stride)
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start = self.windows[index]
        boundary = start + self.history
        frames = self.episodes[episode].frames
        return frames[start:boundary], frames[boundary:boundary + self.horizon], episode
