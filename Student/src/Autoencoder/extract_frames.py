"""Extract 84×84 grayscale frames from Teacher videos into `Student/data/frames/`."""
from __future__ import annotations

import os
from pathlib import Path

from Autoencoder.config import AutoencoderPaths
from videos_to_frames import extract_frames

def run_extract_frames(
    paths: AutoencoderPaths,
    *,
    video_dir: Path | None = None,
    frames_out: Path | None = None,
    skip_if_nonempty: bool = False,
    assume_yes: bool = False,
) -> None:
    from Autoencoder.confirm import confirm_reset

    vdir = Path(video_dir) if video_dir is not None else paths.teacher_videos
    save_dir = Path(frames_out) if frames_out is not None else paths.frames_dir

    print(f"[extract] VIDEO_DIR = {vdir}")
    print(f"[extract] OUT_DIR   = {save_dir}")

    # Ensure directory exists (repo may ship `Teacher/videos/` empty with only .gitkeep)
    vdir.mkdir(parents=True, exist_ok=True)

    save_dir.mkdir(parents=True, exist_ok=True)

    has_mp4 = any(name.endswith(".mp4") for name in os.listdir(vdir))
    if not has_mp4:
        if any(save_dir.iterdir()):
            print(
                f"[extract] SKIP — no .mp4 files in {vdir}; "
                f"using existing frames under {save_dir}"
            )
            return
        raise RuntimeError(
            f"No .mp4 files in {vdir} and {save_dir} is empty. "
            f"Add LunarLander (or other) recordings as .mp4, or copy frames into {save_dir}."
        )

    if skip_if_nonempty and any(save_dir.iterdir()):
        print("[extract] SKIP — frames directory already has files (skip_if_nonempty=True)")
        return

    if any(save_dir.iterdir()):
        if not confirm_reset(
            save_dir,
            "Frames folder already contains data. Remove and re-extract?",
            assume_yes=assume_yes,
        ):
            raise SystemExit(1)

    save_dir.mkdir(parents=True, exist_ok=True)

    extract_frames(vdir, save_dir)
