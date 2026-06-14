#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import cv2


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def student_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def numeric_tail(text: str) -> tuple[int, str]:
    match = re.search(r"(\d+)(?!.*\d)", text)
    return (int(match.group(1)), text) if match else (-1, text)


def extract_frames(video_dir: Path, frames_out: Path) -> None:
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    video_files = sorted(
        [p for p in video_dir.iterdir() if p.suffix == ".mp4"],
        key=lambda p: numeric_tail(p.name),
    )
    if not video_files:
        raise RuntimeError(f"No .mp4 files in {video_dir}")

    if frames_out.exists():
        shutil.rmtree(frames_out)
    frames_out.mkdir(parents=True, exist_ok=True)

    print(f'{"episode":^10} | {"video":^28} | {"frames":^10}')
    print("-" * 56)

    for episode_idx, video_path in enumerate(video_files):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        episode_dir = frames_out / str(episode_idx)
        episode_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (84, 84))
            cv2.imwrite(str(episode_dir / f"{frame_idx}.png"), frame)
            frame_idx += 1

        cap.release()
        print(f"{episode_idx:^10} | {video_path.name[:28]:^28} | {frame_idx:^10}")

    print()
    print(f"[done] videos  : {video_dir}")
    print(f"[done] frames  : {frames_out}")


def main(argv: list[str] | None = None) -> None:
    default_video_dir = repo_root() / "Expert" / "videos"
    default_frames_out = student_dir() / "data" / "frames"

    parser = argparse.ArgumentParser(description="Expert videos -> Student/data/frames")
    parser.add_argument("--video-dir", type=Path, default=default_video_dir)
    parser.add_argument("--frames-out", type=Path, default=default_frames_out)
    args = parser.parse_args(argv)

    print(f"[prep] VIDEO_DIR = {args.video_dir}")
    print(f"[prep] OUT_DIR   = {args.frames_out}")
    extract_frames(args.video_dir, args.frames_out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)